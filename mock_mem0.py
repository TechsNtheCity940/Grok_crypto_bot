"""
Enhanced mock implementation of the mem0 package with in-memory storage
to actually handle trading influence without external dependencies
"""
import uuid
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# In-memory storage for mock implementation
memories_storage = {}
users_storage = set()

class MemoryClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        print("Using enhanced mock MemoryClient with in-memory storage")
    
    def add(self, data, user_id=None, agent_id=None, run_id=None, app_id=None, output_format="v1.0", metadata=None, categories=None):
        """
        Add memory data with simulated storage
        """
        if isinstance(data, str):
            # Handle natural language command for memory deletion
            if "delete all" in data.lower():
                self.delete_all(user_id=user_id)
                return {"status": "success", "message": "All memories deleted based on natural language command"}
            
            # Convert string to message format
            data = [{"role": "user", "content": data}]
        
        # Ensure we have a valid ID for storage
        storage_id = user_id or agent_id or run_id or app_id or "crypto_trader"
        
        # Initialize storage if needed
        if storage_id not in memories_storage:
            memories_storage[storage_id] = []
        
        # Add user to users storage
        if user_id:
            users_storage.add(user_id)
        
        # Create memory entry
        memory_items = []
        for item in data:
            if item.get("role") == "assistant" and agent_id is None:
                continue  # Skip assistant messages unless agent_id is specified
                
            memory_id = str(uuid.uuid4())
            memory_entry = {
                "memory_id": memory_id,
                "text": item.get("content", ""),
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "app_id": app_id,
                "metadata": metadata or {},
                "categories": categories or [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            memories_storage[storage_id].append(memory_entry)
            memory_items.append(memory_entry)
        
        # Format response based on output_format
        if output_format == "v1.1":
            return {
                "status": "success",
                "message": "Memories added successfully",
                "data": memory_items
            }
        else:
            return memory_items
    
    def search(self, query, user_id=None, agent_id=None, run_id=None, app_id=None, 
               categories=None, metadata=None, output_format="v1.0", version="v1", 
               filters=None, limit=10, threshold=0.7):
        """
        Search memories with basic keyword matching for the mock implementation
        """
        storage_id = user_id or agent_id or run_id or app_id or "crypto_trader"
        
        if storage_id not in memories_storage:
            return [] if output_format == "v1.0" else {"status": "success", "data": []}
        
        # Simple keyword search
        results = []
        query_terms = query.lower().split()
        
        for memory in memories_storage[storage_id]:
            # Apply filters if provided
            if filters and version == "v2":
                if not self._apply_filters(memory, filters):
                    continue
            
            # Apply categories filter
            if categories and not any(cat in memory.get("categories", []) for cat in categories):
                continue
                
            # Apply metadata filter
            if metadata:
                memory_metadata = memory.get("metadata", {})
                if not all(memory_metadata.get(k) == v for k, v in metadata.items()):
                    continue
            
            # Basic text matching
            text = memory.get("text", "").lower()
            match_score = sum(1 for term in query_terms if term in text) / len(query_terms) if query_terms else 0
            
            if match_score >= threshold:
                memory_with_score = memory.copy()
                memory_with_score["score"] = match_score
                results.append(memory_with_score)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:limit]
        
        # Format response based on output_format
        if output_format == "v1.1":
            return {
                "status": "success", 
                "message": f"Found {len(results)} memories",
                "data": results
            }
        else:
            return results
    
    def get_all(self, user_id=None, agent_id=None, run_id=None, app_id=None, 
                categories=None, keywords=None, page=1, page_size=100, 
                version="v1", filters=None):
        """
        Get all memories with pagination support
        """
        storage_id = user_id or agent_id or run_id or app_id or "crypto_trader"
        
        if storage_id not in memories_storage:
            return {"page": page, "page_size": page_size, "total": 0, "data": []}
        
        # Filter memories
        results = memories_storage[storage_id].copy()
        
        # Apply filters if provided
        if filters and version == "v2":
            results = [memory for memory in results if self._apply_filters(memory, filters)]
        
        # Apply categories filter
        if categories:
            results = [memory for memory in results 
                      if any(cat in memory.get("categories", []) for cat in categories)]
        
        # Apply keywords filter
        if keywords:
            keywords_lower = keywords.lower()
            results = [memory for memory in results 
                      if keywords_lower in memory.get("text", "").lower()]
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = results[start_idx:end_idx]
        
        return {
            "page": page,
            "page_size": page_size,
            "total": len(results),
            "data": paginated_results
        }
    
    def get(self, memory_id):
        """
        Get a specific memory by ID
        """
        for storage_id, memories in memories_storage.items():
            for memory in memories:
                if memory.get("memory_id") == memory_id:
                    return memory
        return None
    
    def users(self):
        """
        Get all users with memories
        """
        return list(users_storage)
    
    def history(self, memory_id):
        """
        Get history of memory changes
        """
        # In this mock implementation, we don't track history
        # Just return the current memory if it exists
        memory = self.get(memory_id)
        if memory:
            return [memory]
        return []
    
    def update(self, memory_id, text):
        """
        Update a memory with new text
        """
        for storage_id, memories in memories_storage.items():
            for i, memory in enumerate(memories):
                if memory.get("memory_id") == memory_id:
                    memories[i]["text"] = text
                    memories[i]["updated_at"] = datetime.now().isoformat()
                    return {"status": "success", "message": "Memory updated successfully"}
        return {"status": "error", "message": "Memory not found"}
    
    def delete(self, memory_id):
        """
        Delete a specific memory
        """
        for storage_id, memories in memories_storage.items():
            for i, memory in enumerate(memories):
                if memory.get("memory_id") == memory_id:
                    del memories[i]
                    return {"status": "success", "message": "Memory deleted successfully"}
        return {"status": "error", "message": "Memory not found"}
    
    def delete_all(self, user_id=None, agent_id=None, run_id=None, app_id=None):
        """
        Delete all memories for a specific ID
        """
        storage_id = user_id or agent_id or run_id or app_id
        if storage_id and storage_id in memories_storage:
            memories_storage[storage_id] = []
            return {"status": "success", "message": f"All memories for {storage_id} deleted successfully"}
        return {"status": "error", "message": "ID not found"}
    
    def delete_users(self, user_id=None, agent_id=None, app_id=None, run_id=None):
        """
        Delete specific users
        """
        if user_id and user_id in users_storage:
            users_storage.remove(user_id)
            if user_id in memories_storage:
                del memories_storage[user_id]
            return {"status": "success", "message": f"User {user_id} deleted successfully"}
        
        # For other IDs, just clear their memory storage if they exist
        storage_id = agent_id or app_id or run_id
        if storage_id and storage_id in memories_storage:
            del memories_storage[storage_id]
            return {"status": "success", "message": f"Storage for {storage_id} deleted successfully"}
            
        return {"status": "error", "message": "ID not found"}
    
    def reset(self):
        """
        Reset all memories and users
        """
        global memories_storage, users_storage
        memories_storage = {}
        users_storage = set()
        return {"status": "success", "message": "Client reset successfully"}
    
    def batch_update(self, updates):
        """
        Batch update memories
        """
        results = []
        for update in updates:
            memory_id = update.get("memory_id")
            text = update.get("text")
            if memory_id and text:
                result = self.update(memory_id, text)
                results.append(result)
        return {"status": "success", "results": results}
    
    def batch_delete(self, deletes):
        """
        Batch delete memories
        """
        results = []
        for delete in deletes:
            memory_id = delete.get("memory_id")
            if memory_id:
                result = self.delete(memory_id)
                results.append(result)
        return {"status": "success", "results": results}
    
    def _apply_filters(self, memory, filters):
        """
        Apply complex filters to a memory
        """
        if "AND" in filters:
            return all(self._apply_filter(memory, subfilter) for subfilter in filters["AND"])
        elif "OR" in filters:
            return any(self._apply_filter(memory, subfilter) for subfilter in filters["OR"])
        else:
            return self._apply_filter(memory, filters)
    
    def _apply_filter(self, memory, filter_item):
        """
        Apply a single filter to a memory
        """
        for key, condition in filter_item.items():
            if key not in memory and key != "metadata" and key != "categories":
                return False
                
            # Handle metadata filtering
            if key == "metadata":
                memory_metadata = memory.get("metadata", {})
                for meta_key, meta_value in condition.items():
                    if meta_key not in memory_metadata or memory_metadata[meta_key] != meta_value:
                        return False
                return True
                
            # Handle categories filtering
            if key == "categories":
                memory_categories = memory.get("categories", [])
                if isinstance(condition, dict) and "contains" in condition:
                    return condition["contains"] in memory_categories
                return False
                
            # Handle date filtering
            if key in ["created_at", "updated_at"]:
                memory_date = memory.get(key, "")
                if isinstance(condition, dict):
                    if "gte" in condition and memory_date < condition["gte"]:
                        return False
                    if "lte" in condition and memory_date > condition["lte"]:
                        return False
                    return True
                    
            # Handle simple equality
            if memory.get(key) != condition:
                return False
                
        return True


class AsyncMemoryClient(MemoryClient):
    """
    Mock implementation of AsyncMemoryClient that simply calls the
    synchronous methods from MemoryClient
    """
    async def add(self, data, user_id=None, agent_id=None, run_id=None, app_id=None, output_format="v1.0", metadata=None, categories=None):
        return super().add(data, user_id, agent_id, run_id, app_id, output_format, metadata, categories)
    
    async def search(self, query, user_id=None, agent_id=None, run_id=None, app_id=None, 
                    categories=None, metadata=None, output_format="v1.0", version="v1", 
                    filters=None, limit=10, threshold=0.7):
        return super().search(query, user_id, agent_id, run_id, app_id, 
                              categories, metadata, output_format, version, 
                              filters, limit, threshold)
    
    async def get_all(self, user_id=None, agent_id=None, run_id=None, app_id=None, 
                     categories=None, keywords=None, page=1, page_size=100, 
                     version="v1", filters=None):
        return super().get_all(user_id, agent_id, run_id, app_id, 
                              categories, keywords, page, page_size, 
                              version, filters)
