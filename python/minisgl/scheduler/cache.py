from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str, page_size: int = 1):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages
        self.page_size = page_size

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        total_tokens = self.manager.size_info.evictable_size + len(self._free_slots) * self.page_size
        return total_tokens // self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict((needed_len - free_len) * self.page_size)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        start_page = old_handle.cached_len // self.page_size
        end_page = in_cache_len // self.page_size
        self._free(indices[start_page : end_page])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size // self.page_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size // self.page_size}) != num_pages({self.num_pages})"
            )
