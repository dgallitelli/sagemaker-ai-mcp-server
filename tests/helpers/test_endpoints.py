"""Tests for SageMaker AI Endpoints and Endpoint Configurations."""

import pytest
from sagemaker_ai_mcp_server.helpers.endpoints import (
    delete_endpoint,
    delete_endpoint_config,
    describe_endpoint,
    describe_endpoint_config,
    list_endpoint_configs,
    list_endpoints,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_list_endpoints(mock_get_sagemaker_client):
    """Test listing SageMaker AI Endpoints."""
    mock_client = MagicMock()
    mock_client.list_endpoints.return_value = {'Endpoints': [{'EndpointName': 'test-endpoint'}]}
    mock_get_sagemaker_client.return_value = mock_client
    endpoints = await list_endpoints()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_endpoints.assert_called_once()
    assert endpoints == [{'EndpointName': 'test-endpoint'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_list_endpoint_configs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Endpoint Configurations."""
    mock_client = MagicMock()
    mock_client.list_endpoint_configs.return_value = {
        'EndpointConfigs': [{'EndpointConfigName': 'test-config'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    configs = await list_endpoint_configs()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_endpoint_configs.assert_called_once()
    assert configs == [{'EndpointConfigName': 'test-config'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_describe_endpoint(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Endpoint."""
    mock_client = MagicMock()
    expected_response = {'EndpointName': 'test-endpoint', 'Status': 'InService'}
    mock_client.describe_endpoint.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_endpoint('test-endpoint')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_endpoint.assert_called_once_with(EndpointName='test-endpoint')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_describe_endpoint_config(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Endpoint Config."""
    mock_client = MagicMock()
    expected_response = {'EndpointConfigName': 'test-config', 'ProductionVariants': []}
    mock_client.describe_endpoint_config.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_endpoint_config('test-config')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_endpoint_config.assert_called_once_with(EndpointConfigName='test-config')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_delete_endpoint(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Endpoint."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_endpoint('test-endpoint')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_endpoint.assert_called_once_with(EndpointName='test-endpoint')


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.endpoints.get_sagemaker_client')
async def test_delete_endpoint_config(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Endpoint Config."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_endpoint_config('test-config')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_endpoint_config.assert_called_once_with(EndpointConfigName='test-config')
