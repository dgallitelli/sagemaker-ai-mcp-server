"""Tests for SageMaker AI Model Cards."""

import pytest
from sagemaker_ai_mcp_server.helpers.model_cards import (
    delete_model_card,
    describe_model_card,
    list_model_card_export_jobs,
    list_model_card_versions,
    list_model_cards,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.model_cards.get_sagemaker_client')
async def test_list_model_cards(mock_get_sagemaker_client):
    """Test listing SageMaker AI Model Cards."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'ModelCardSummaries': [{'ModelCardName': 'test-card', 'ModelCardArn': 'arn:aws:...'}]
    }
    mock_client.list_model_cards.return_value = mock_response
    cards = await list_model_cards()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_model_cards.assert_called_once()
    expected = [{'ModelCardName': 'test-card', 'ModelCardArn': 'arn:aws:...'}]
    assert cards == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.model_cards.get_sagemaker_client')
async def test_list_model_card_export_jobs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Model Card Export Jobs."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'ModelCardExportJobSummaries': [
            {'ModelCardExportJobName': 'test-export-job', 'ModelCardArn': 'arn:aws:...'}
        ]
    }
    mock_client.list_model_card_export_jobs.return_value = mock_response
    jobs = await list_model_card_export_jobs()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_model_card_export_jobs.assert_called_once()
    expected = [{'ModelCardExportJobName': 'test-export-job', 'ModelCardArn': 'arn:aws:...'}]
    assert jobs == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.model_cards.get_sagemaker_client')
async def test_list_model_card_versions(mock_get_sagemaker_client):
    """Test listing SageMaker AI Model Card Versions."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'ModelCardVersionSummaryList': [{'ModelCardVersion': '1.0', 'ModelCardArn': 'arn:aws:...'}]
    }
    mock_client.list_model_card_versions.return_value = mock_response
    versions = await list_model_card_versions('test-card')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_model_card_versions.assert_called_once_with(ModelCardName='test-card')
    expected = [{'ModelCardVersion': '1.0', 'ModelCardArn': 'arn:aws:...'}]
    assert versions == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.model_cards.get_sagemaker_client')
async def test_describe_model_card(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Model Card."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {
        'ModelCardName': 'test-card',
        'ModelCardArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-card/test-card',
        'ModelCardStatus': 'Draft',
    }
    mock_client.describe_model_card.return_value = expected_response
    response = await describe_model_card('test-card')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_model_card.assert_called_once_with(ModelCardName='test-card')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.model_cards.get_sagemaker_client')
async def test_delete_model_card(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Model Card."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_model_card('test-card')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_model_card.assert_called_once_with(ModelCardName='test-card')
