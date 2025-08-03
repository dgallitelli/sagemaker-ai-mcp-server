"""Tests for SageMaker AI Models."""

import pytest
from sagemaker_ai_mcp_server.helpers.models import delete_model, describe_model, list_models
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.models.get_sagemaker_client')
async def test_list_models(mock_get_sagemaker_client):
    """Test listing SageMaker AI Models."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'Models': [{'ModelName': 'test-model', 'CreationTime': '2023-01-01T00:00:00Z'}]
    }
    mock_client.list_models.return_value = mock_response
    models = await list_models()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_models.assert_called_once()
    expected = [{'ModelName': 'test-model', 'CreationTime': '2023-01-01T00:00:00Z'}]
    assert models == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.models.get_sagemaker_client')
async def test_describe_model(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Model."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {
        'ModelName': 'test-model',
        'PrimaryContainer': {
            'Image': '123456789012.dkr.ecr.us-west-2.amazonaws.com/test-image:latest'
        },
        'ExecutionRoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    }
    mock_client.describe_model.return_value = expected_response
    response = await describe_model('test-model')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_model.assert_called_once_with(ModelName='test-model')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.models.get_sagemaker_client')
async def test_delete_model(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Model."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_model('test-model')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_model.assert_called_once_with(ModelName='test-model')
