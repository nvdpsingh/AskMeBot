"""
MIRIX Memory System - Modular Memory Framework for LLM-based AI Agents

A comprehensive multi-agent memory system with six specialized memory types:
1. Core Memory - High-priority persistent information
2. Episodic Memory - Time-stamped events and experiences  
3. Semantic Memory - Abstract facts and knowledge
4. Procedural Memory - Step-by-step instructions and workflows
5. Resource Memory - Documents and multimodal files
6. Knowledge Vault - Sensitive and verbatim data

Each memory type is managed by a dedicated agent, coordinated by a Meta Memory Manager.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Enumeration of MIRIX memory types"""
    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    RESOURCE = "resource"
    KNOWLEDGE_VAULT = "knowledge_vault"

@dataclass
class MemoryEntry:
    """Base class for all memory entries"""
    id: str
    content: str
    timestamp: datetime
    priority: int = 1  # 1-10 scale, 10 being highest
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class CoreMemoryEntry(MemoryEntry):
    """Core memory for high-priority persistent information"""
    user_id: str = "default"
    agent_persona: str = ""
    importance_score: float = 1.0

@dataclass
class EpisodicMemoryEntry(MemoryEntry):
    """Episodic memory for time-stamped events and experiences"""
    event_type: str = "general"
    location: str = ""
    participants: List[str] = None
    emotional_context: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if self.participants is None:
            self.participants = []

@dataclass
class SemanticMemoryEntry(MemoryEntry):
    """Semantic memory for abstract facts and knowledge"""
    concept: str = ""
    relationships: List[str] = None
    confidence_score: float = 1.0
    source: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if self.relationships is None:
            self.relationships = []

@dataclass
class ProceduralMemoryEntry(MemoryEntry):
    """Procedural memory for step-by-step instructions"""
    task_name: str = ""
    steps: List[str] = None
    prerequisites: List[str] = None
    success_criteria: List[str] = None
    difficulty_level: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if self.steps is None:
            self.steps = []
        if self.prerequisites is None:
            self.prerequisites = []
        if self.success_criteria is None:
            self.success_criteria = []

@dataclass
class ResourceMemoryEntry(MemoryEntry):
    """Resource memory for documents and multimodal files"""
    file_path: str = ""
    file_type: str = ""
    file_size: int = 0
    content_hash: str = ""
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.last_accessed is None:
            self.last_accessed = datetime.now(timezone.utc)

@dataclass
class KnowledgeVaultEntry(MemoryEntry):
    """Knowledge vault for sensitive and verbatim data"""
    data_type: str = "text"  # text, credentials, contact, etc.
    encryption_level: int = 1  # 1-5 scale
    access_restrictions: List[str] = None
    retention_period: Optional[datetime] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.access_restrictions is None:
            self.access_restrictions = []

class MemoryAgent:
    """Base class for all memory agents"""
    
    def __init__(self, memory_type: MemoryType, storage_path: str):
        self.memory_type = memory_type
        self.storage_path = Path(storage_path) / memory_type.value
        self.entries: Dict[str, MemoryEntry] = {}
        self.read_only_mode = False
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.load_entries()
        except (OSError, PermissionError) as e:
            logger.warning(f"Read-only file system detected for {memory_type.value} memory. Using in-memory storage only.")
            self.read_only_mode = True
    
    def load_entries(self):
        """Load memory entries from storage"""
        if self.read_only_mode:
            logger.info(f"In-memory mode: Starting with empty {self.memory_type.value} memory")
            return
            
        try:
            entries_file = self.storage_path / "entries.json"
            if entries_file.exists():
                # Use synchronous file operations to avoid event loop issues
                with open(entries_file, 'r') as f:
                    data = json.load(f)
                    for entry_id, entry_data in data.items():
                        self.entries[entry_id] = self._deserialize_entry(entry_data)
                    
                logger.info(f"Loaded {len(self.entries)} entries for {self.memory_type.value} memory")
        except Exception as e:
            logger.error(f"Error loading {self.memory_type.value} memory: {e}")
            # Fall back to read-only mode
            self.read_only_mode = True
            logger.warning(f"Switched to read-only mode for {self.memory_type.value} memory")
    
    def save_entries(self):
        """Save memory entries to storage"""
        if self.read_only_mode:
            logger.info(f"In-memory mode: {len(self.entries)} entries for {self.memory_type.value} memory (not persisted)")
            return
            
        try:
            entries_file = self.storage_path / "entries.json"
            data = {entry_id: self._serialize_entry(entry) 
                   for entry_id, entry in self.entries.items()}
            
            # Use synchronous file operations to avoid event loop issues
            with open(entries_file, 'w') as f:
                f.write(json.dumps(data, indent=2, default=str))
                
            logger.info(f"Saved {len(self.entries)} entries for {self.memory_type.value} memory")
        except Exception as e:
            logger.error(f"Error saving {self.memory_type.value} memory: {e}")
            # Fall back to read-only mode
            self.read_only_mode = True
            logger.warning(f"Switched to read-only mode for {self.memory_type.value} memory")
    
    def _serialize_entry(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Serialize memory entry to dictionary"""
        data = asdict(entry)
        data['timestamp'] = entry.timestamp.isoformat()
        if hasattr(entry, 'last_accessed') and entry.last_accessed:
            data['last_accessed'] = entry.last_accessed.isoformat()
        return data
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> MemoryEntry:
        """Deserialize dictionary to memory entry"""
        # This will be overridden by each specific memory agent
        raise NotImplementedError
    
    def add_entry(self, entry: MemoryEntry) -> str:
        """Add a new memory entry"""
        entry.id = self._generate_id(entry)
        self.entries[entry.id] = entry
        self.save_entries()
        logger.info(f"Added {self.memory_type.value} memory entry: {entry.id}")
        return entry.id
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry"""
        return self.entries.get(entry_id)
    
    def search_entries(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries based on query"""
        # Simple text-based search - can be enhanced with vector search
        query_lower = query.lower()
        results = []
        
        for entry in self.entries.values():
            score = 0
            if query_lower in entry.content.lower():
                score += 10
            if query_lower in ' '.join(entry.tags).lower():
                score += 5
            if hasattr(entry, 'concept') and query_lower in entry.concept.lower():
                score += 8
            if hasattr(entry, 'task_name') and query_lower in entry.task_name.lower():
                score += 8
            
            # Search in metadata for episodic memories
            if hasattr(entry, 'metadata') and entry.metadata:
                metadata_text = ' '.join(str(v) for v in entry.metadata.values() if isinstance(v, str))
                if query_lower in metadata_text.lower():
                    score += 7  # High score for metadata matches
                
            if score > 0:
                results.append((entry, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in results[:limit]]
    
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory entry"""
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        self.save_entries()
        logger.info(f"Updated {self.memory_type.value} memory entry: {entry_id}")
        return True
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.save_entries()
            logger.info(f"Deleted {self.memory_type.value} memory entry: {entry_id}")
            return True
        return False
    
    def _generate_id(self, entry: MemoryEntry) -> str:
        """Generate unique ID for memory entry"""
        content_hash = hashlib.md5(entry.content.encode()).hexdigest()[:8]
        timestamp_str = entry.timestamp.strftime("%Y%m%d%H%M%S")
        return f"{self.memory_type.value}_{timestamp_str}_{content_hash}"

class CoreMemoryAgent(MemoryAgent):
    """Agent for managing core memory (high-priority persistent information)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.CORE, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> CoreMemoryEntry:
        """Deserialize to CoreMemoryEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return CoreMemoryEntry(**data)
    
    def add_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Add or update user profile in core memory"""
        content = f"User Profile: {json.dumps(profile_data, indent=2)}"
        entry = CoreMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=10,  # Highest priority
            user_id=user_id,
            tags=["user_profile", "persistent"],
            metadata=profile_data
        )
        return self.add_entry(entry)
    
    def add_agent_persona(self, persona_data: Dict[str, Any]) -> str:
        """Add or update agent persona in core memory"""
        content = f"Agent Persona: {json.dumps(persona_data, indent=2)}"
        entry = CoreMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=9,
            agent_persona=persona_data.get("name", "AskMeBot"),
            tags=["agent_persona", "persistent"],
            metadata=persona_data
        )
        return self.add_entry(entry)

class EpisodicMemoryAgent(MemoryAgent):
    """Agent for managing episodic memory (time-stamped events and experiences)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.EPISODIC, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> EpisodicMemoryEntry:
        """Deserialize to EpisodicMemoryEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return EpisodicMemoryEntry(**data)
    
    def add_conversation_event(self, user_id: str, conversation_data: Dict[str, Any]) -> str:
        """Add conversation event to episodic memory"""
        content = f"Conversation: {conversation_data.get('summary', 'User interaction')}"
        entry = EpisodicMemoryEntry(
            id="",  # Will be auto-generated
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=conversation_data.get('priority', 5),
            event_type="conversation",
            participants=[user_id, "AskMeBot"],
            tags=["conversation", "interaction"],
            metadata=conversation_data
        )
        return self.add_entry(entry)
    
    def add_task_completion(self, user_id: str, task_data: Dict[str, Any]) -> str:
        """Add task completion event to episodic memory"""
        content = f"Task Completed: {task_data.get('task_name', 'Unknown task')}"
        entry = EpisodicMemoryEntry(
            id="",  # Will be auto-generated
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=task_data.get('priority', 6),
            event_type="task_completion",
            participants=[user_id],
            tags=["task", "completion"],
            metadata=task_data
        )
        return self.add_entry(entry)

class SemanticMemoryAgent(MemoryAgent):
    """Agent for managing semantic memory (abstract facts and knowledge)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.SEMANTIC, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> SemanticMemoryEntry:
        """Deserialize to SemanticMemoryEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return SemanticMemoryEntry(**data)
    
    def add_fact(self, concept: str, fact: str, source: str = "", confidence: float = 1.0) -> str:
        """Add a fact to semantic memory"""
        entry = SemanticMemoryEntry(
            content=fact,
            timestamp=datetime.now(timezone.utc),
            priority=7,
            concept=concept,
            confidence_score=confidence,
            source=source,
            tags=["fact", "knowledge"],
            metadata={"concept": concept, "source": source}
        )
        return self.add_entry(entry)
    
    def add_relationship(self, entity1: str, relationship: str, entity2: str, context: str = "") -> str:
        """Add a relationship to semantic memory"""
        content = f"{entity1} {relationship} {entity2}. Context: {context}"
        entry = SemanticMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=6,
            concept=f"{entity1}-{relationship}-{entity2}",
            relationships=[entity1, entity2],
            tags=["relationship", "knowledge"],
            metadata={"entity1": entity1, "relationship": relationship, "entity2": entity2}
        )
        return self.add_entry(entry)

class ProceduralMemoryAgent(MemoryAgent):
    """Agent for managing procedural memory (step-by-step instructions and workflows)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.PROCEDURAL, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> ProceduralMemoryEntry:
        """Deserialize to ProceduralMemoryEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return ProceduralMemoryEntry(**data)
    
    def add_workflow(self, task_name: str, steps: List[str], prerequisites: List[str] = None, 
                    success_criteria: List[str] = None, difficulty: int = 1) -> str:
        """Add a workflow to procedural memory"""
        content = f"Workflow: {task_name}\nSteps: {'; '.join(steps)}"
        entry = ProceduralMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=8,
            task_name=task_name,
            steps=steps,
            prerequisites=prerequisites or [],
            success_criteria=success_criteria or [],
            difficulty_level=difficulty,
            tags=["workflow", "procedure"],
            metadata={"task_name": task_name, "difficulty": difficulty}
        )
        return self.add_entry(entry)
    
    def add_instruction(self, instruction: str, category: str = "general") -> str:
        """Add an instruction to procedural memory"""
        entry = ProceduralMemoryEntry(
            content=instruction,
            timestamp=datetime.now(timezone.utc),
            priority=5,
            task_name=category,
            steps=[instruction],
            tags=["instruction", category],
            metadata={"category": category}
        )
        return self.add_entry(entry)

class ResourceMemoryAgent(MemoryAgent):
    """Agent for managing resource memory (documents and multimodal files)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.RESOURCE, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> ResourceMemoryEntry:
        """Deserialize to ResourceMemoryEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'last_accessed' in data and data['last_accessed']:
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return ResourceMemoryEntry(**data)
    
    def add_document(self, file_path: str, content: str, file_type: str = "text") -> str:
        """Add a document to resource memory"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        entry = ResourceMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=6,
            file_path=file_path,
            file_type=file_type,
            file_size=len(content),
            content_hash=content_hash,
            tags=["document", file_type],
            metadata={"file_path": file_path, "file_type": file_type}
        )
        return self.add_entry(entry)
    
    def add_multimodal_content(self, content: str, content_type: str, metadata: Dict[str, Any]) -> str:
        """Add multimodal content to resource memory"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        entry = ResourceMemoryEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=5,
            file_type=content_type,
            content_hash=content_hash,
            tags=["multimodal", content_type],
            metadata=metadata
        )
        return self.add_entry(entry)

class KnowledgeVaultAgent(MemoryAgent):
    """Agent for managing knowledge vault (sensitive and verbatim data)"""
    
    def __init__(self, storage_path: str):
        super().__init__(MemoryType.KNOWLEDGE_VAULT, storage_path)
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> KnowledgeVaultEntry:
        """Deserialize to KnowledgeVaultEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'retention_period' in data and data['retention_period']:
            data['retention_period'] = datetime.fromisoformat(data['retention_period'])
        return KnowledgeVaultEntry(**data)
    
    def add_credential(self, service: str, credential_data: Dict[str, Any], 
                      encryption_level: int = 3) -> str:
        """Add credential to knowledge vault"""
        content = f"Credential for {service}: [ENCRYPTED]"
        entry = KnowledgeVaultEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=9,
            data_type="credential",
            encryption_level=encryption_level,
            access_restrictions=["admin"],
            tags=["credential", service, "sensitive"],
            metadata={"service": service, "encrypted": True}
        )
        return self.add_entry(entry)
    
    def add_contact(self, contact_data: Dict[str, Any]) -> str:
        """Add contact information to knowledge vault"""
        content = f"Contact: {contact_data.get('name', 'Unknown')}"
        entry = KnowledgeVaultEntry(
            content=content,
            timestamp=datetime.now(timezone.utc),
            priority=7,
            data_type="contact",
            encryption_level=2,
            tags=["contact", "personal"],
            metadata=contact_data
        )
        return self.add_entry(entry)

class MetaMemoryManager:
    """Meta Memory Manager to coordinate all memory agents"""
    
    def __init__(self, storage_path: str = "memory_data"):
        self.storage_path = storage_path
        self.agents = {
            MemoryType.CORE: CoreMemoryAgent(storage_path),
            MemoryType.EPISODIC: EpisodicMemoryAgent(storage_path),
            MemoryType.SEMANTIC: SemanticMemoryAgent(storage_path),
            MemoryType.PROCEDURAL: ProceduralMemoryAgent(storage_path),
            MemoryType.RESOURCE: ResourceMemoryAgent(storage_path),
            MemoryType.KNOWLEDGE_VAULT: KnowledgeVaultAgent(storage_path)
        }
        logger.info("MIRIX Meta Memory Manager initialized with all agents")
    
    def add_memory(self, memory_type: MemoryType, entry: MemoryEntry) -> str:
        """Add memory entry to specific agent"""
        return self.agents[memory_type].add_entry(entry)
    
    def search_memory(self, query: str, memory_types: List[MemoryType] = None, 
                     limit_per_type: int = 5) -> Dict[MemoryType, List[MemoryEntry]]:
        """Search across multiple memory types"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        results = {}
        for memory_type in memory_types:
            if memory_type in self.agents:
                results[memory_type] = self.agents[memory_type].search_entries(query, limit_per_type)
        
        return results
    
    def get_relevant_memories(self, user_query: str, context: Dict[str, Any] = None) -> List[MemoryEntry]:
        """Active retrieval mechanism - automatically infer relevant memories"""
        logger.info(f"🔍 MIRIX: Searching for relevant memories for query: {user_query[:100]}...")
        
        # Determine which memory types to search based on query analysis
        memory_types_to_search = self._analyze_query_intent(user_query)
        
        # Search across relevant memory types
        search_results = self.search_memory(user_query, memory_types_to_search, limit_per_type=3)
        
        # Combine and rank results
        all_results = []
        for memory_type, entries in search_results.items():
            for entry in entries:
                # Add memory type context to entry
                entry.metadata = entry.metadata or {}
                entry.metadata['memory_type'] = memory_type.value
                all_results.append(entry)
        
        # Sort by priority and relevance
        all_results.sort(key=lambda x: (x.priority, x.timestamp), reverse=True)
        
        logger.info(f"🎯 MIRIX: Found {len(all_results)} relevant memories")
        return all_results[:10]  # Return top 10 most relevant
    
    def _analyze_query_intent(self, query: str) -> List[MemoryType]:
        """Analyze query to determine which memory types to search"""
        query_lower = query.lower()
        memory_types = []
        
        # Core memory indicators
        if any(word in query_lower for word in ['who am i', 'my profile', 'about me', 'persona']):
            memory_types.append(MemoryType.CORE)
        
        # Episodic memory indicators - expanded to catch more conversation references
        if any(word in query_lower for word in ['remember', 'happened', 'last time', 'conversation', 'event', 
                                               'discuss', 'talked', 'mentioned', 'said', 'told', 'asked',
                                               'earlier', 'before', 'previous', 'past', 'we talked', 'we discussed']):
            memory_types.append(MemoryType.EPISODIC)
        
        # Semantic memory indicators - expanded to catch more knowledge queries
        if any(word in query_lower for word in ['what is', 'define', 'explain', 'fact', 'knowledge', 'about',
                                               'tell me about', 'information', 'details', 'describe']):
            memory_types.append(MemoryType.SEMANTIC)
        
        # Procedural memory indicators
        if any(word in query_lower for word in ['how to', 'steps', 'process', 'workflow', 'procedure']):
            memory_types.append(MemoryType.PROCEDURAL)
        
        # Resource memory indicators
        if any(word in query_lower for word in ['document', 'file', 'image', 'video', 'resource']):
            memory_types.append(MemoryType.RESOURCE)
        
        # Knowledge vault indicators
        if any(word in query_lower for word in ['password', 'credential', 'contact', 'sensitive']):
            memory_types.append(MemoryType.KNOWLEDGE_VAULT)
        
        # If no specific indicators, search all types
        if not memory_types:
            memory_types = list(MemoryType)
        
        return memory_types
    
    def get_memory_context(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """Get formatted memory context for LLM prompt"""
        relevant_memories = self.get_relevant_memories(user_query, context)
        
        if not relevant_memories:
            return ""
        
        context_parts = ["🧠 **MIRIX Memory Context:**"]
        
        for memory in relevant_memories:
            memory_type = memory.metadata.get('memory_type', 'unknown')
            context_parts.append(f"\n**{memory_type.upper()} Memory:**")
            context_parts.append(f"- {memory.content}")
            if memory.tags:
                context_parts.append(f"  Tags: {', '.join(memory.tags)}")
        
        return "\n".join(context_parts)
    
    def update_conversation_memory(self, user_id: str, conversation_data: Dict[str, Any]):
        """Update episodic memory with conversation data"""
        self.agents[MemoryType.EPISODIC].add_conversation_event(user_id, conversation_data)
    
    def save_all_memories(self):
        """Save all memory agents"""
        for agent in self.agents.values():
            agent.save_entries()
        logger.info("💾 MIRIX: All memories saved")

# Global MIRIX instance
mirix_manager = MetaMemoryManager()
