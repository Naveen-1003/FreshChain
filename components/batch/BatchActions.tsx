import React, { useState, useEffect } from 'react';
import { Package, RefreshCw, AlertTriangle, MapPin, Clock, CheckCircle, Wallet, ExternalLink } from 'lucide-react';
import { getAuthService, UserRole } from '../../lib/services/authService';
import { getBackendService, BatchDetails, BatchStage } from '../../lib/services/backendService';
import { getMetaMaskService } from '../../lib/services/metamaskService';

const BatchActions: React.FC = () => {
  const [batches, setBatches] = useState<BatchDetails[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  // Update stage form
  const [selectedBatch, setSelectedBatch] = useState<string>('');
  const [newStage, setNewStage] = useState<BatchStage>(BatchStage.HARVESTED);
  const [newLocation, setNewLocation] = useState('');
  
  // MetaMask integration
  const [isProcessingTx, setIsProcessingTx] = useState(false);
  const [currentTxHash, setCurrentTxHash] = useState('');

  const authService = getAuthService();
  const backendService = getBackendService();
  const metamaskService = getMetaMaskService();
  const user = authService.getUser();

  useEffect(() => {
    loadBatches();
  }, []);

  const loadBatches = async () => {
    setIsLoading(true);
    try {
      // Use different endpoint based on user role
      const response = user?.role === UserRole.ADMIN 
        ? await backendService.getAllBatches()
        : await backendService.getBatchesByStage();
      
      if (response.success && response.data) {
        setBatches(response.data);
      } else {
        setError(response.error || 'Failed to load batches');
      }
    } catch (error) {
      setError('Network error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const updateBatchStage = async () => {
    if (!selectedBatch || !newStage || !newLocation) {
      setError('Please fill in all fields');
      return;
    }

    setIsProcessingTx(true);
    setError('');
    setSuccess('');

    try {
      if (user?.role === UserRole.ADMIN) {
        // Admin updates via MetaMask (backend handles the transaction)
        const response = await backendService.adminUpdateStageWithMetaMask({
          batchId: selectedBatch,
          stage: newStage,
          location: newLocation
        });

        if (response.success) {
          setSuccess(`Batch updated successfully! TX: ${response.transactionHash}`);
          setSelectedBatch('');
          setNewStage(BatchStage.HARVESTED);
          setNewLocation('');
          await loadBatches();
        } else {
          throw new Error(response.error || 'Failed to update batch');
        }
      } else {
        // Non-admin users use regular API
        const response = await backendService.updateBatchStage({
          batchId: selectedBatch,
          stage: newStage,
          location: newLocation
        });

        if (response.success) {
          setSuccess(`Batch updated successfully! TX: ${response.transactionHash}`);
          setSelectedBatch('');
          setNewStage(BatchStage.HARVESTED);
          setNewLocation('');
          await loadBatches();
        } else {
          throw new Error(response.error || 'Failed to update batch');
        }
      }
    } catch (error: any) {
      setError(error.message);
    } finally {
      setIsProcessingTx(false);
    }
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'Created': return 'bg-gray-100 text-gray-800';
      case 'Harvested': return 'bg-green-100 text-green-800';
      case 'In Transit': return 'bg-blue-100 text-blue-800';
      case 'At Retailer': return 'bg-purple-100 text-purple-800';
      case 'Selling': return 'bg-emerald-100 text-emerald-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const canUpdateStage = user?.role === UserRole.ADMIN || user?.role === UserRole.RETAILER || user?.role === UserRole.TRANSPORTER;

  return (
    <div className="space-y-6">
      {/* Header with Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Package className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Batches Created</p>
              <p className="text-2xl font-bold text-gray-900">{batches.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">In Transit</p>
              <p className="text-2xl font-bold text-gray-900">
                {batches.filter(b => b.currentStage === 'In Transit').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <MapPin className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">At Retailer</p>
              <p className="text-2xl font-bold text-gray-900">
                {batches.filter(b => b.currentStage === 'At Retailer').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="h-5 w-5 text-emerald-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Alerts Triggered</p>
              <p className="text-2xl font-bold text-gray-900">
                {batches.reduce((acc, b) => acc + b.alerts.length, 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4">
        <button
          onClick={loadBatches}
          disabled={isLoading}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>

        {user?.role === UserRole.ADMIN && (
          <button className="flex items-center space-x-2 px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors">
            <Wallet className="h-4 w-4" />
            <span>Report Alert</span>
          </button>
        )}
      </div>

      {/* Transaction Processing Status */}
      {isProcessingTx && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />
            <div>
              <p className="font-medium text-blue-800">Processing Transaction</p>
              <p className="text-sm text-blue-600">
                {user?.role === UserRole.ADMIN ? 'Please confirm in MetaMask...' : 'Submitting to blockchain...'}
              </p>
              {currentTxHash && (
                <p className="text-xs text-blue-500 font-mono mt-1">TX: {currentTxHash}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Success/Error Messages */}
      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <p className="text-green-800">{success}</p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5 text-red-600" />
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}

      {/* Update Stage Form */}
      {canUpdateStage && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Update Batch Stage</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Batch</label>
              <select
                value={selectedBatch}
                onChange={(e) => setSelectedBatch(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              >
                <option value="">Choose batch...</option>
                {batches.filter(b => !b.isFinalStage).map(batch => (
                  <option key={batch.batchId} value={batch.batchId}>
                    {batch.batchId} - {batch.productType}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">New Stage</label>
              <select
                value={newStage}
                onChange={(e) => setNewStage(e.target.value as BatchStage)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              >
                <option value={BatchStage.HARVESTED}>Harvested</option>
                <option value={BatchStage.IN_TRANSIT}>In Transit</option>
                <option value={BatchStage.AT_RETAILER}>At Retailer</option>
                <option value={BatchStage.SELLING}>Selling</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Location</label>
              <input
                type="text"
                value={newLocation}
                onChange={(e) => setNewLocation(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                placeholder="e.g., Distribution Center"
              />
            </div>
          </div>
          <div className="flex space-x-3 mt-4">
            <button
              onClick={updateBatchStage}
              disabled={isProcessingTx}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              {user?.role === UserRole.ADMIN && <Wallet className="h-4 w-4" />}
              <span>{isProcessingTx ? 'Processing...' : 'Update Stage'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Batch List */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Manage Current Batches</h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Batch ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Product Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Stage
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {batches.map((batch) => (
                <tr key={batch.batchId} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="font-medium text-gray-900">{batch.batchId}</div>
                    <div className="text-sm text-gray-500">{new Date(batch.created).toLocaleDateString()}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{batch.productType}</div>
                    <div className="text-sm text-gray-500">{batch.currentLocation}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStageColor(batch.currentStage)}`}>
                      {batch.currentStage}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${batch.isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                      <span className="text-sm text-gray-600">
                        {batch.isFinalStage ? 'Final' : 'Active'}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    {!batch.isFinalStage && canUpdateStage && (
                      <button
                        onClick={() => {
                          setSelectedBatch(batch.batchId);
                          setNewLocation(batch.currentLocation);
                        }}
                        className="text-emerald-600 hover:text-emerald-900 mr-3"
                      >
                        Update Stage
                      </button>
                    )}
                    {batch.locationHistory && batch.locationHistory[0] && batch.locationHistory[0].transactionHash ? (
                      batch.locationHistory[0].transactionHash.startsWith('0x') && batch.locationHistory[0].transactionHash.length === 66 ? (
                        <a
                          href={`https://explorer-mezame.shardeum.org/tx/${batch.locationHistory[0].transactionHash}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-900 inline-flex items-center space-x-1"
                          title="View transaction on Shardeum Explorer"
                        >
                          <ExternalLink className="h-3 w-3" />
                          <span>View TX</span>
                        </a>
                      ) : (
                        <span 
                          className="text-gray-400 inline-flex items-center space-x-1 cursor-not-allowed"
                          title="Demo transaction - not available on explorer"
                        >
                          <ExternalLink className="h-3 w-3" />
                          <span>Demo TX</span>
                        </span>
                      )
                    ) : (
                      <span className="text-gray-400 text-sm">No TX</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {batches.length === 0 && !isLoading && (
            <div className="text-center py-8">
              <Package className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No batches found. Create your first batch to get started.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchActions;