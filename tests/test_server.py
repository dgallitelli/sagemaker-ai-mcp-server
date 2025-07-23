"""Tests for the server functions in the SageMaker AI MCP Server."""

import pytest
from sagemaker_ai_mcp_server.server import (
    delete_endpoint_config_sagemaker,
    delete_endpoint_sagemaker,
    describe_endpoint_config_sagemaker,
    describe_endpoint_sagemaker,
    list_endpoint_configs_sagemaker,
    list_endpoints_sagemaker,
)
from unittest.mock import patch


@pytest.mark.asyncio
async def test_list_endpoints_sagemaker():
    """Test the list_endpoints_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_endpoints'
    ) as mock_list_endpoints:
        mock_list_endpoints.return_value = [{'EndpointName': 'test-endpoint'}]

        result = await list_endpoints_sagemaker()

        mock_list_endpoints.assert_called_once()
        assert result == {'endpoints': [{'EndpointName': 'test-endpoint'}]}


@pytest.mark.asyncio
async def test_list_endpoint_configs_sagemaker():
    """Test the list_endpoint_configs_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_endpoint_configs'
    ) as mock_list_configs:
        mock_list_configs.return_value = [
            {'EndpointConfigName': 'test-config'}
        ]

        result = await list_endpoint_configs_sagemaker()

        mock_list_configs.assert_called_once()
        assert result == {
            'endpoint_configs': [{'EndpointConfigName': 'test-config'}]
        }


@pytest.mark.asyncio
async def test_delete_endpoint_sagemaker():
    """Test the delete_endpoint_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.delete_endpoint'
    ) as mock_delete_endpoint:
        endpoint_name = 'test-endpoint'
        result = await delete_endpoint_sagemaker(endpoint_name)

        mock_delete_endpoint.assert_called_once_with(endpoint_name)
        expected_msg = f"Endpoint '{endpoint_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_delete_endpoint_config_sagemaker():
    """Test the delete_endpoint_config_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.delete_endpoint_config'
    ) as mock_delete_config:
        config_name = 'test-endpoint-config'

        result = await delete_endpoint_config_sagemaker(config_name)

        mock_delete_config.assert_called_once_with(config_name)
        expected_msg = f"Endpoint Config '{config_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_describe_endpoint_sagemaker():
    """Test the describe_endpoint_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_endpoint'
    ) as mock_describe_endpoint:
        endpoint_name = 'test-endpoint'
        expected_result = {
            'EndpointName': endpoint_name,
            'EndpointStatus': 'InService',
            'CreationTime': '2023-01-01T00:00:00'
        }
        mock_describe_endpoint.return_value = expected_result

        result = await describe_endpoint_sagemaker(endpoint_name)

        mock_describe_endpoint.assert_called_once_with(endpoint_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_describe_endpoint_config_sagemaker():
    """Test the describe_endpoint_config_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_endpoint_config'
    ) as mock_describe_config:
        config_name = 'test-endpoint-config'
        expected_result = {
            'EndpointConfigName': config_name,
            'CreationTime': '2023-01-01T00:00:00',
            'ProductionVariants': [{'VariantName': 'test-variant'}]
        }
        mock_describe_config.return_value = expected_result

        result = await describe_endpoint_config_sagemaker(config_name)

        mock_describe_config.assert_called_once_with(config_name)
        assert result == expected_result
