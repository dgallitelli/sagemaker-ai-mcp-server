"""Tests for SageMaker AI Domains."""

import pytest
from sagemaker_ai_mcp_server.helpers.domains import (
    create_presigned_domain_url,
    delete_domain,
    describe_domain,
    list_domains,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.domains.get_sagemaker_client')
async def test_list_domains(mock_get_sagemaker_client):
    """Test listing SageMaker AI Domains."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {'Domains': [{'DomainId': 'test-domain', 'DomainName': 'Test Domain'}]}
    mock_client.list_domains.return_value = mock_response
    domains = await list_domains()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_domains.assert_called_once()
    expected = [{'DomainId': 'test-domain', 'DomainName': 'Test Domain'}]
    assert domains == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.domains.get_sagemaker_client')
async def test_create_presigned_domain_url(mock_get_sagemaker_client):
    """Test creating a presigned domain URL."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {'AuthorizedUrl': 'https://example.com/presigned-domain-url'}
    mock_client.create_presigned_domain_url.return_value = expected_response
    url = await create_presigned_domain_url('test-domain', 'test-profile-name')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.create_presigned_domain_url.assert_called_once_with(
        DomainId='test-domain', UserProfileName='test-profile-name', ExpirationSeconds=3600
    )
    assert url == 'https://example.com/presigned-domain-url'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.domains.get_sagemaker_client')
async def test_describe_domain(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Domain."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {
        'DomainId': 'test-domain',
        'DomainName': 'Test Domain',
        'Status': 'InService',
    }
    mock_client.describe_domain.return_value = expected_response
    response = await describe_domain('test-domain')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_domain.assert_called_once_with(DomainId='test-domain')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.domains.get_sagemaker_client')
async def test_delete_domain(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Domain."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_domain('test-domain')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_domain.assert_called_once_with(DomainId='test-domain')
