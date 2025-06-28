"""Database operations using ChromaDB."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings


class Database:
    """Handles database operations for the application."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the ChromaDB database. Uses settings.CHROMA_DB_PATH if None.
        """
        self.db_path = db_path or settings.CHROMA_DB_PATH
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        self._init_collections()
    
    def _init_collections(self):
        """Initialize the required collections if they don't exist."""
        self.books = self.client.get_or_create_collection(
            "books",
            metadata={"description": "Stores book metadata and content"}
        )
        
        self.chapters = self.client.get_or_create_collection(
            "chapters",
            metadata={"description": "Stores chapter content and metadata"}
        )
        
        self.versions = self.client.get_or_create_collection(
            "versions",
            metadata={"description": "Tracks versions of chapters and books"}
        )
    
    def add_book(self, title: str, metadata: Optional[Dict] = None) -> str:
        """Add a new book to the database.
        
        Args:
            title: Title of the book
            metadata: Additional metadata about the book
            
        Returns:
            str: The ID of the created book
        """
        book_id = f"book_{uuid.uuid4().hex}"
        
        book_data = {
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "draft",
            **(metadata or {})
        }
        
        self.books.add(
            documents=[json.dumps(book_data)],
            metadatas=[{"type": "book"}],
            ids=[book_id]
        )
        
        logger.info(f"Added new book: {title} (ID: {book_id})")
        return book_id
    
    def add_chapter(
        self,
        book_id: str,
        title: str,
        content: str,
        chapter_number: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new chapter to the database.
        
        Args:
            book_id: ID of the book this chapter belongs to
            title: Title of the chapter
            content: Text content of the chapter
            chapter_number: Optional chapter number
            metadata: Additional metadata about the chapter
            
        Returns:
            str: The ID of the created chapter
        """
        chapter_id = f"chapter_{uuid.uuid4().hex}"
        
        chapter_data = {
            "book_id": book_id,
            "title": title,
            "content": content,
            "chapter_number": chapter_number,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "draft",
            **(metadata or {})
        }
        
        # Store the chapter data in the metadata
        self.chapters.add(
            documents=[content],
            metadatas=[{
                "type": "chapter",
                "book_id": book_id,
                "data": json.dumps(chapter_data)
            }],
            ids=[chapter_id]
        )
        
        # Create initial version
        self.create_version(chapter_id, content, "Initial version")
        
        logger.info(f"Added new chapter: {title} (ID: {chapter_id})")
        return chapter_id
    
    def create_version(
        self,
        content_id: str,
        content: str,
        notes: str,
        author: str = "system",
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new version of a chapter or book.
        
        Args:
            content_id: ID of the content being versioned
            content: The content of this version
            notes: Notes about this version
            author: Who created this version
            metadata: Additional metadata
            
        Returns:
            str: The version ID
        """
        version_id = f"version_{uuid.uuid4().hex}"
        
        version_data = {
            "content_id": content_id,
            "content": content,
            "notes": notes,
            "author": author,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        self.versions.add(
            documents=[content],
            metadatas=[{
                "type": "version",
                "content_id": content_id,
                "data": json.dumps(version_data)
            }],
            ids=[version_id]
        )
        
        logger.info(f"Created new version for {content_id} (Version ID: {version_id})")
        return version_id
    
    def get_chapter(self, chapter_id: str) -> Optional[Dict]:
        """Retrieve a chapter by its ID.
        
        Args:
            chapter_id: The ID of the chapter to retrieve
            
        Returns:
            Optional[Dict]: The chapter data, or None if not found
        """
        try:
            result = self.chapters.get(
                ids=[chapter_id],
                include=["metadatas", "documents"]
            )
            
            if not result["ids"]:
                return None
                
            # Extract metadata from the first result
            metadata = json.loads(result["metadatas"][0].get("data", "{}"))
            return {
                "id": chapter_id,
                "content": result["documents"][0],
                **metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving chapter {chapter_id}: {e}")
            return None
    
    def get_versions(self, content_id: str) -> List[Dict]:
        """Get all versions for a specific content item.
        
        Args:
            content_id: ID of the content to get versions for
            
        Returns:
            List[Dict]: List of version data
        """
        try:
            results = self.versions.query(
                query_texts=[""],  # Empty query to get all
                where={"content_id": content_id},
                include=["metadatas", "documents"]
            )
            
            versions = []
            for idx, version_id in enumerate(results["ids"][0]):
                metadata = json.loads(results["metadatas"][0][idx].get("data", "{}"))
                versions.append({
                    "id": version_id,
                    "content": results["documents"][0][idx],
                    **metadata
                })
                
            return versions
        except Exception as e:
            logger.error(f"Error retrieving versions for {content_id}: {e}")
            return []
    
    def search_chapters(self, query: str, book_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search for chapters matching a query.
        
        Args:
            query: The search query
            book_id: Optional book ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: List of matching chapters with scores
        """
        try:
            where_clause = {"type": "chapter"}
            if book_id:
                where_clause["book_id"] = book_id
                
            results = self.chapters.query(
                query_texts=[query],
                where=where_clause,
                n_results=min(limit, 10),
                include=["metadatas", "documents", "distances"]
            )
            
            matches = []
            for idx, chapter_id in enumerate(results["ids"][0]):
                metadata = json.loads(results["metadatas"][0][idx].get("data", "{}"))
                matches.append({
                    "id": chapter_id,
                    "score": 1.0 - results["distances"][0][idx] if results["distances"] and results["distances"][0] else 0.0,
                    "title": metadata.get("title", ""),
                    "book_id": metadata.get("book_id", ""),
                    "content_preview": (results["documents"][0][idx][:200] + "...") if results["documents"] and results["documents"][0] else ""
                })
                
            return matches
        except Exception as e:
            logger.error(f"Error searching chapters: {e}")
            return []
    
    def update_chapter(
        self,
        chapter_id: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update a chapter's content and/or metadata.
        
        Args:
            chapter_id: ID of the chapter to update
            content: New content (if updating)
            title: New title (if updating)
            status: New status (if updating)
            metadata: Additional metadata to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Get existing chapter data
            chapter = self.get_chapter(chapter_id)
            if not chapter:
                logger.error(f"Chapter {chapter_id} not found")
                return False
            
            # Update fields if provided
            update_data = {}
            if content is not None:
                update_data["content"] = content
            if title is not None:
                update_data["title"] = title
            if status is not None:
                update_data["status"] = status
            
            # Update metadata
            chapter_metadata = chapter.get("metadata", {})
            if metadata:
                chapter_metadata.update(metadata)
            
            # Update the document in the collection
            self.chapters.update(
                ids=[chapter_id],
                documents=[content] if content is not None else None,
                metadatas=[{
                    **chapter_metadata,
                    "data": json.dumps({
                        **chapter,
                        **update_data,
                        "metadata": chapter_metadata,
                        "updated_at": datetime.utcnow().isoformat()
                    })
                }]
            )
            
            logger.info(f"Updated chapter {chapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chapter {chapter_id}: {e}")
            return False
