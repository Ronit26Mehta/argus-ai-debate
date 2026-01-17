"""
Unit tests for ARGUS MCP Integration module.
"""

import pytest
import json

from argus.mcp.config import MCPServerConfig, MCPClientConfig, TransportType, MCPToolSchema
from argus.mcp.server import ArgusServer, MCPRequest, MCPResponse
from argus.mcp.client import MCPClient
from argus.mcp.tools import ToolAdapter, ToolSchemaGenerator, MCPToolWrapper
from argus.mcp.resources import ResourceRegistry, ConfigResource


class TestMCPConfig:
    """Tests for MCP configuration."""
    
    def test_default_server_config(self):
        config = MCPServerConfig()
        assert config.name == "argus-mcp-server"
        assert config.transport == TransportType.STDIO
        assert config.enable_tools is True
    
    def test_default_client_config(self):
        config = MCPClientConfig()
        assert config.transport == TransportType.STDIO
        assert config.timeout == 30.0


class TestArgusServer:
    """Tests for MCP server."""
    
    def test_server_initialization(self):
        server = ArgusServer(name="test-server")
        assert server.config.name == "test-server"
    
    def test_register_tool_decorator(self):
        server = ArgusServer()
        
        @server.tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b
        
        assert "add" in server._tools
        assert "add" in server._tool_schemas
    
    def test_register_resource_decorator(self):
        server = ArgusServer()
        
        @server.resource("test://config")
        def get_config() -> dict:
            return {"setting": "value"}
        
        assert "test://config" in server._resources
    
    def test_handle_initialize(self):
        server = ArgusServer()
        request = MCPRequest(id=1, method="initialize", params={"protocolVersion": "2024-11-05"})
        response = server.handle_request(request)
        assert response.result is not None
        assert "serverInfo" in response.result
    
    def test_handle_tools_list(self):
        server = ArgusServer()
        
        @server.tool()
        def test_tool(x: str) -> str:
            return x
        
        request = MCPRequest(id=1, method="tools/list", params={})
        response = server.handle_request(request)
        assert response.result is not None
        assert len(response.result["tools"]) == 1
    
    def test_handle_tools_call(self):
        server = ArgusServer()
        
        @server.tool()
        def multiply(a: int, b: int) -> int:
            return a * b
        
        request = MCPRequest(
            id=1, method="tools/call",
            params={"name": "multiply", "arguments": {"a": 3, "b": 4}}
        )
        response = server.handle_request(request)
        assert response.result is not None
        content = json.loads(response.result["content"][0]["text"])
        assert content == 12
    
    def test_handle_unknown_method(self):
        server = ArgusServer()
        request = MCPRequest(id=1, method="unknown/method", params={})
        response = server.handle_request(request)
        assert response.error is not None
        assert response.error["code"] == -32601


class TestMCPClient:
    """Tests for MCP client."""
    
    def test_client_initialization(self):
        client = MCPClient()
        assert client.is_connected is False
    
    def test_client_not_connected(self):
        client = MCPClient()
        tools = client.list_tools()
        assert tools == []


class TestToolAdapter:
    """Tests for tool adapters."""
    
    def test_schema_generator(self):
        def my_func(name: str, count: int, active: bool = True) -> dict:
            """A test function."""
            return {}
        
        schema = ToolSchemaGenerator.from_function(my_func)
        assert schema.name == "my_func"
        assert "name" in schema.input_schema["properties"]
        assert "count" in schema.input_schema["properties"]
        assert "name" in schema.input_schema["required"]
        assert "active" not in schema.input_schema["required"]
    
    def test_schema_types(self):
        def typed_func(s: str, i: int, f: float, b: bool) -> None:
            pass
        
        schema = ToolSchemaGenerator.from_function(typed_func)
        props = schema.input_schema["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"


class TestResourceRegistry:
    """Tests for resource registry."""
    
    def test_register_resource(self):
        registry = ResourceRegistry()
        resource = ConfigResource()
        registry.register(resource)
        assert registry.get(resource.uri) is not None
    
    def test_list_all_resources(self):
        registry = ResourceRegistry()
        resource = ConfigResource()
        registry.register(resource)
        schemas = registry.list_all()
        assert len(schemas) == 1
        assert schemas[0].uri == "argus://config/current"
