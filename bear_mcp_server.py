#!/usr/bin/env python3
"""
Bear Notes MCP Server

A Model Context Protocol server for searching and accessing Bear notes data.
Provides flexible search capabilities by directly accessing Bear's SQLite database.
"""

import asyncio
import json
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import traceback
import subprocess
import urllib.parse
import logging

# Configure logging to minimize sensitive data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

# Create a logger for the Bear MCP server
logger = logging.getLogger("bear_mcp")

# Function to sanitize note data for logging
def sanitize_note_for_logging(note_data):
    """Remove sensitive content from note data before logging"""
    if not note_data:
        return note_data
        
    sanitized = note_data.copy()
    if 'content' in sanitized:
        sanitized['content'] = f"<content redacted, {len(sanitized['content'])} chars>"
    if 'preview' in sanitized:
        sanitized['preview'] = "<preview redacted>"
    return sanitized

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions  # Import from lowlevel
import mcp.server.stdio
import mcp.types as types

# Add error handling to help with debugging
try:
    class BearDatabase:
        """Handle Bear database operations"""
        
        def __init__(self):
            self.db_path = Path.home() / "Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite"
            self._validate_database()
        
        def _validate_database(self):
            """Validate that Bear database exists and is accessible"""
            if not self.db_path.exists():
                raise FileNotFoundError(f"Bear database not found at {self.db_path}")
            
            if not os.access(self.db_path, os.R_OK):
                raise PermissionError(f"Cannot read Bear database at {self.db_path}")
        
        def _get_connection(self) -> sqlite3.Connection:
            """Get a read-only connection to Bear database"""
            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        
        @staticmethod
        def _core_data_timestamp_to_datetime(timestamp: float) -> datetime:
            """Convert Core Data timestamp to Python datetime"""
            # Core Data uses seconds since 2001-01-01 00:00:00 GMT
            core_data_epoch = datetime(2001, 1, 1)
            return core_data_epoch + timedelta(seconds=timestamp)
        
        def search_notes(self, query: str = "", tags: List[str] = None, 
                        limit: int = 50, include_trashed: bool = False) -> List[Dict[str, Any]]:
            """
            Search Bear notes with flexible filtering
            
            Args:
                query: Search term for title and content
                tags: List of tags to filter by
                limit: Maximum number of results
                include_trashed: Whether to include trashed notes
            """
            with self._get_connection() as conn:
                # Base query
                sql_parts = [
                    "SELECT DISTINCT n.Z_PK as id, n.ZTITLE as title, n.ZTEXT as content,",
                    "n.ZCREATIONDATE as creation_date, n.ZMODIFICATIONDATE as modification_date,",
                    "n.ZTRASHED as trashed"
                ]
                
                from_parts = ["FROM ZSFNOTE n"]
                where_parts = []
                params = []
                
                # Handle trashed notes filter
                if not include_trashed:
                    where_parts.append("n.ZTRASHED = 0")
                
                # Handle text search
                if query:
                    where_parts.append("(n.ZTITLE LIKE ? OR n.ZTEXT LIKE ?)")
                    search_pattern = f"%{query}%"
                    params.extend([search_pattern, search_pattern])
                
                # Handle tag filtering
                if tags:
                    from_parts.extend([
                        "JOIN ZSFNOTETAG nt ON n.Z_PK = nt.ZNOTE",
                        "JOIN ZSFTAG t ON nt.ZTAG = t.Z_PK"
                    ])
                    
                    if len(tags) == 1:
                        where_parts.append("t.ZTITLE = ?")
                        params.append(tags[0])
                    else:
                        placeholders = ",".join("?" * len(tags))
                        where_parts.append(f"t.ZTITLE IN ({placeholders})")
                        params.extend(tags)
                
                # Construct final query
                sql = " ".join(sql_parts + from_parts)
                if where_parts:
                    sql += " WHERE " + " AND ".join(where_parts)
                
                sql += " ORDER BY n.ZMODIFICATIONDATE DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                results = []
                
                for row in cursor:
                    # Convert timestamps and clean up data
                    creation_date = self._core_data_timestamp_to_datetime(row['creation_date']) if row['creation_date'] else None
                    modification_date = self._core_data_timestamp_to_datetime(row['modification_date']) if row['modification_date'] else None
                    
                    # Extract preview from content (first 200 chars, clean of markdown)
                    content = row['content'] or ""
                    preview = re.sub(r'[#*`\[\]()_~]', '', content)[:200].strip()
                    if len(content) > 200:
                        preview += "..."
                    
                    results.append({
                        'id': row['id'],
                        'title': row['title'] or "Untitled",
                        'content': content,
                        'preview': preview,
                        'creation_date': creation_date.isoformat() if creation_date else None,
                        'modification_date': modification_date.isoformat() if modification_date else None,
                        'trashed': bool(row['trashed'])
                    })
                
                return results
        
        def get_note_by_id(self, note_id: int) -> Optional[Dict[str, Any]]:
            """Get a specific note by its ID"""
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT Z_PK as id, ZTITLE as title, ZTEXT as content,
                           ZCREATIONDATE as creation_date, ZMODIFICATIONDATE as modification_date,
                           ZTRASHED as trashed
                    FROM ZSFNOTE 
                    WHERE Z_PK = ?
                    """, 
                    (note_id,)
                )
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                creation_date = self._core_data_timestamp_to_datetime(row['creation_date']) if row['creation_date'] else None
                modification_date = self._core_data_timestamp_to_datetime(row['modification_date']) if row['modification_date'] else None
                
                return {
                    'id': row['id'],
                    'title': row['title'] or "Untitled",
                    'content': row['content'] or "",
                    'creation_date': creation_date.isoformat() if creation_date else None,
                    'modification_date': modification_date.isoformat() if modification_date else None,
                    'trashed': bool(row['trashed'])
                }
        
        def list_tags(self) -> List[Dict[str, Any]]:
            """Get all available tags"""
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT t.Z_PK as id, t.ZTITLE as title, COUNT(nt.ZNOTE) as note_count
                    FROM ZSFTAG t
                    LEFT JOIN ZSFNOTETAG nt ON t.Z_PK = nt.ZTAG
                    LEFT JOIN ZSFNOTE n ON nt.ZNOTE = n.Z_PK AND n.ZTRASHED = 0
                    GROUP BY t.Z_PK, t.ZTITLE
                    ORDER BY t.ZTITLE
                    """
                )
                
                return [
                    {
                        'id': row['id'],
                        'title': row['title'],
                        'note_count': row['note_count'] or 0
                    }
                    for row in cursor
                ]
        
        def search_by_tag(self, tag: str, limit: int = 50) -> List[Dict[str, Any]]:
            """Search notes by a specific tag"""
            return self.search_notes(tags=[tag], limit=limit)

        def get_note_identifiers(self, note_id: int) -> Dict[str, Any]:
            """Get all possible identifiers for a note"""
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT Z_PK, ZUNIQUEIDENTIFIER, ZTITLE
                    FROM ZSFNOTE 
                    WHERE Z_PK = ?
                    """, 
                    (note_id,)
                )
                
                row = cursor.fetchone()
                if not row:
                    return {}
                
                return {
                    'Z_PK': row['Z_PK'],
                    'ZUNIQUEIDENTIFIER': row['ZUNIQUEIDENTIFIER'],
                    'ZTITLE': row['ZTITLE']
                }

        def open_note_in_bear(self, note_id: int) -> bool:
            """Open a specific note in the Bear application in a new window"""
            try:
                # Get the note to verify it exists
                note = self.get_note_by_id(note_id)
                if not note:
                    logger.error(f"Note with ID {note_id} not found")
                    return False
                
                # Log only the title and ID, not the content
                logger.info(f"Opening note: '{note['title']}' (ID: {note_id})")
                
                # Get the unique identifier
                identifiers = self.get_note_identifiers(note_id)
                
                # Create the Bear URL with new_window parameter
                bear_url = f"bear://x-callback-url/open-note?id={note_id}&new_window=yes"
                logger.debug(f"Opening Bear with URL: {bear_url}")
                
                # Use the 'open' command to launch Bear with the URL
                result = subprocess.run(["open", bear_url], check=True, capture_output=True)
                logger.debug(f"Open command result: {result.returncode}")
                
                # If the first attempt doesn't work, try with the unique identifier
                if 'ZUNIQUEIDENTIFIER' in identifiers and identifiers['ZUNIQUEIDENTIFIER']:
                    unique_id = identifiers['ZUNIQUEIDENTIFIER']
                    bear_url_alt = f"bear://x-callback-url/open-note?id={unique_id}&new_window=yes"
                    logger.debug(f"Trying alternative URL with unique ID: {bear_url_alt}")
                    
                    result_alt = subprocess.run(["open", bear_url_alt], check=True, capture_output=True)
                    logger.debug(f"Alternative open command result: {result_alt.returncode}")
                
                return True
            except Exception as e:
                logger.error(f"Error opening note in Bear: {str(e)}")
                logger.debug(traceback.format_exc())
                return False


    # Initialize the MCP server
    server = Server("bear-notes")
    bear_db = BearDatabase()


    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available tools for the Bear MCP server"""
        return [
            types.Tool(
                name="search_bear_notes",
                description="Search Bear notes by content, title, or tags",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term to look for in note titles and content"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific tags"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 50)",
                            "default": 50
                        },
                        "include_trashed": {
                            "type": "boolean",
                            "description": "Include trashed notes in results (default: false)",
                            "default": False
                        }
                    }
                }
            ),
            types.Tool(
                name="get_bear_note",
                description="Get a specific Bear note by its ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "The unique ID of the note to retrieve"
                        }
                    },
                    "required": ["note_id"]
                }
            ),
            types.Tool(
                name="list_bear_tags",
                description="List all available tags in Bear with note counts",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="search_bear_by_tag",
                description="Find all notes with a specific tag",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "The tag to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 50)",
                            "default": 50
                        }
                    },
                    "required": ["tag"]
                }
            ),
            types.Tool(
                name="open_bear_note",
                description="Open a specific Bear note in the Bear application",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "The unique ID of the note to open"
                        }
                    },
                    "required": ["note_id"]
                }
            )
        ]


    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        """Handle tool calls for Bear operations"""
        
        try:
            if name == "search_bear_notes":
                query = arguments.get("query", "")
                tags = arguments.get("tags", [])
                limit = arguments.get("limit", 50)
                include_trashed = arguments.get("include_trashed", False)
                
                logger.info(f"Searching notes with query: '{query}', tags: {tags}, limit: {limit}")
                
                results = bear_db.search_notes(
                    query=query, 
                    tags=tags, 
                    limit=limit, 
                    include_trashed=include_trashed
                )
                
                # Sanitize results for logging
                sanitized_results = [sanitize_note_for_logging(note) for note in results]
                logger.debug(f"Found {len(results)} notes")
                
                # Return full results to the client
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "results": results,
                        "count": len(results),
                        "search_params": {
                            "query": query,
                            "tags": tags,
                            "limit": limit,
                            "include_trashed": include_trashed
                        }
                    }, indent=2)
                )]
            
            elif name == "get_bear_note":
                note_id = arguments["note_id"]
                logger.info(f"Getting note with ID: {note_id}")
                
                note = bear_db.get_note_by_id(note_id)
                
                if not note:
                    logger.warning(f"Note with ID {note_id} not found")
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({"error": f"Note with ID {note_id} not found"})
                    )]
                
                # Log sanitized version
                sanitized_note = sanitize_note_for_logging(note)
                logger.debug(f"Retrieved note: {sanitized_note}")
                
                # Return full note to client
                return [types.TextContent(
                    type="text",
                    text=json.dumps(note, indent=2)
                )]
            
            elif name == "list_bear_tags":
                tags = bear_db.list_tags()
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "tags": tags,
                        "count": len(tags)
                    }, indent=2)
                )]
            
            elif name == "search_bear_by_tag":
                tag = arguments["tag"]
                limit = arguments.get("limit", 50)
                
                results = bear_db.search_by_tag(tag, limit)
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "results": results,
                        "count": len(results),
                        "tag": tag
                    }, indent=2)
                )]
            
            elif name == "open_bear_note":
                note_id = arguments["note_id"]
                logger.info(f"Opening note with ID: {note_id}")
                
                # Get the note details first
                note = bear_db.get_note_by_id(note_id)
                if not note:
                    logger.warning(f"Note with ID {note_id} not found")
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": f"Note with ID {note_id} not found"
                        }, indent=2)
                    )]
                
                # Log sanitized version
                sanitized_note = sanitize_note_for_logging(note)
                logger.debug(f"Found note: {sanitized_note}")
                
                # Get identifiers for debugging
                identifiers = bear_db.get_note_identifiers(note_id)
                
                success = bear_db.open_note_in_bear(note_id)
                
                if success:
                    logger.info(f"Successfully opened note '{note['title']}' (ID: {note_id})")
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "message": f"Opened note '{note['title']}' (ID: {note_id}) in a new Bear window",
                            "note_details": {
                                "title": note['title'],
                                "id": note_id
                            }
                        }, indent=2)
                    )]
                else:
                    logger.error(f"Failed to open note '{note['title']}' (ID: {note_id})")
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": f"Failed to open note '{note['title']}' (ID: {note_id}) in Bear",
                            "note_details": {
                                "title": note['title'],
                                "id": note_id
                            }
                        }, indent=2)
                    )]
            
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"})
                )]
        
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]


    async def main():
        """Run the Bear MCP server"""
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                # Create a minimal ServerCapabilities object
                from mcp.types import ServerCapabilities, ToolsCapability
                
                # Create tools capability
                tools_capability = ToolsCapability(
                    enabled=True
                )
                
                # Create server capabilities
                capabilities = ServerCapabilities(
                    tools=tools_capability
                )
                
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="bear-notes",
                        server_version="1.0.0",
                        capabilities=capabilities
                    )
                )
        except Exception as e:
            print(f"Error in MCP server: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise


    if __name__ == "__main__":
        asyncio.run(main())
except Exception as e:
    print(f"Fatal error in Bear MCP server: {str(e)}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
