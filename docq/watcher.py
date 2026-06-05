"""Filesystem watcher using watchdog. Updates index live when docs are added/changed/deleted."""

from __future__ import annotations
import time
from pathlib import Path
from threading import Thread, Event
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .indexer import Indexer


class _DocHandler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer, on_change: Optional[Callable[[str, Path], None]] = None):
        super().__init__()
        self.indexer = indexer
        self.on_change = on_change

    def _handle(self, event: FileSystemEvent, is_dir: bool = False):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if not self.indexer._is_supported(p):  # type: ignore[attr-defined]
            return
        # Small debounce for rapid saves
        time.sleep(0.15)
        try:
            if event.event_type in ("created", "modified"):
                changed = self.indexer.add_or_update_file(p)
                if changed and self.on_change:
                    self.on_change("updated", p)
            elif event.event_type == "deleted":
                self.indexer.remove_file(p)
                if self.on_change:
                    self.on_change("deleted", p)
        except Exception:
            pass  # never kill watcher thread

    def on_created(self, event: FileSystemEvent):
        self._handle(event)

    def on_modified(self, event: FileSystemEvent):
        self._handle(event)

    def on_deleted(self, event: FileSystemEvent):
        self._handle(event)

    # moved is delete+create in practice for most editors
    def on_moved(self, event: FileSystemEvent):
        if hasattr(event, "dest_path"):
            # treat as update on dest
            self._handle(type("E", (), {"src_path": event.dest_path, "is_directory": False, "event_type": "modified"})())  # type: ignore
        self._handle(event)  # also clean old if needed


class FolderWatcher:
    def __init__(self, indexer: Indexer, on_change: Optional[Callable[[str, Path], None]] = None):
        self.indexer = indexer
        self.on_change = on_change
        self.observer: Optional[Observer] = None
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self, recursive: bool = True) -> None:
        if self.observer:
            return
        handler = _DocHandler(self.indexer, self.on_change)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.indexer.cfg.folder), recursive=recursive)
        self.observer.start()

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2)
            self.observer = None

    def run_forever(self) -> None:
        """Blocking. For use in dedicated watcher process."""
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop()

    def run_in_thread(self) -> Thread:
        """Non-blocking watcher (used by interactive CLI)."""
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        t = Thread(target=self.run_forever, daemon=True, name="docq-watcher")
        self._thread = t
        t.start()
        return t

    def stop_thread(self) -> None:
        self._stop_event.set()
        self.stop()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
