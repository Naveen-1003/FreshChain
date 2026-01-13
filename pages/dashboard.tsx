import { NextPage } from 'next';
import { useState, useEffect, useMemo, useCallback } from 'react';
import { useRouter } from 'next/router';
import { 
  Package, 
  Eye, 
  Shield, 
  Zap, 
  Camera, 
  LogOut, 
  User, 
  Truck, 
  BarChart3, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  MapPin,
  RefreshCw,
  ExternalLink,
  QrCode,
  Wallet,
  ShoppingBag,
  Download,
  X
} from 'lucide-react';
import { getAuthService, UserRole } from '../lib/services/authService';
import DashboardLayout from '../components/layout/DashboardLayout';
import AdminWallet from '../components/admin/AdminWallet';
import { getBackendService, BatchDetails } from '../lib/services/backendService';

const DashboardPage: NextPage = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [batches, setBatches] = useState<BatchDetails[]>([]);
  const [recentActivity, setRecentActivity] = useState<any[]>([]);
  const [showQRModal, setShowQRModal] = useState(false);
  const [selectedBatchForQR, setSelectedBatchForQR] = useState<BatchDetails | null>(null);
  const router = useRouter();
  const authService = getAuthService();
  const backendService = getBackendService();
  const user = authService.getUser();

  useEffect(() => {
    let mounted = true;
    
    if (!authService.isAuthenticated()) {
      router.push('/login');
    } else {
      setIsLoading(false);
      if (mounted) {
        loadDashboardData();
      }
    }
    
    return () => { mounted = false; };
  }, []);  // Empty dependency array - only run once on mount

  // Helper functions - MUST be defined before loadDashboardData
  const getStageProgress = (stage: string) => {
    switch (stage) {
      case 'Created': return 20;
      case 'Harvested': return 40;
      case 'In Transit': return 60;
      case 'At Retailer': return 80;
      case 'Selling': return 100;
      default: return 0;
    }
  };

  const getDisplayStage = (stage: string, userRole: string | undefined) => {
    // For transporters, show "Reached Retailer" instead of "At Retailer"
    if (userRole === 'transporter' && stage === 'At Retailer') {
      return 'Reached Retailer';
    }
    return stage;
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'Created': return 'text-gray-600';
      case 'Harvested': return 'text-green-600';
      case 'In Transit': return 'text-blue-600';
      case 'At Retailer': return 'text-purple-600';
      case 'Selling': return 'text-emerald-600';
      default: return 'text-gray-600';
    }
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'Created': return Package;
      case 'Harvested': return CheckCircle;
      case 'In Transit': return Truck;
      case 'At Retailer': return MapPin;
      case 'Selling': return Eye;
      default: return Package;
    }
  };

  const getStageBackground = (stage: string) => {
    switch (stage) {
      case 'Created': return 'bg-gray-50 border-l-4 border-gray-400';
      case 'Harvested': return 'bg-green-50 border-l-4 border-green-400';
      case 'In Transit': return 'bg-blue-50 border-l-4 border-blue-400';
      case 'At Retailer': return 'bg-purple-50 border-l-4 border-purple-400';
      case 'Selling': return 'bg-emerald-50 border-l-4 border-emerald-400';
      default: return 'bg-gray-50 border-l-4 border-gray-400';
    }
  };

  const loadDashboardData = useCallback(async () => {
    try {
      let response;
      if (user?.role === UserRole.ADMIN) {
        response = await backendService.getAllBatches();
      } else if (user?.role === UserRole.TRANSPORTER) {
        response = await backendService.getBatchesByStage('In Transit');
      } else if (user?.role === UserRole.RETAILER) {
        response = await backendService.getBatchesByStage('At Retailer');
      } else if (user?.role === UserRole.CONSUMER) {
        response = await backendService.getBatchesByStage('Selling');
      } else {
        response = await backendService.getBatchesByStage();
      }

      if (response.success && Array.isArray(response.data)) {
        const fetchedBatches = response.data as BatchDetails[];
        setBatches(fetchedBatches);

        const activity = fetchedBatches
          .flatMap((batch) => {
            const historyItems = Array.isArray(batch.locationHistory) ? batch.locationHistory : [];
            return historyItems.map((history, index) => {
              const stageName = history.stage || batch.currentStage || 'Created';
              return {
                id: `${batch.batchId}-${history.timestamp}-${index}`,
                batchId: batch.batchId,
                productType: batch.productType,
                action: stageName,
                location: history.location,
                timestamp: history.timestamp,
                updatedBy: history.updatedBy,
                stageIcon: getStageIcon(stageName),
                stageColor: getStageColor(stageName),
                stageBackground: getStageBackground(stageName),
                isLatest: index === 0,
                stageProgress: getStageProgress(stageName),
              };
            });
          })
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, 5);

        setRecentActivity(activity);
      } else {
        setBatches([]);
        setRecentActivity([]);
      }
    } catch (error) {
      console.error('Dashboard load error:', error);
      setBatches([]);
      setRecentActivity([]);
    }
  }, [backendService, user?.role]);

  const generateQRCode = (batch: BatchDetails) => {
    setSelectedBatchForQR(batch);
    setShowQRModal(true);
  };

  const downloadQRCode = () => {
    if (!selectedBatchForQR) return;
    
    // QR code contains only the batch ID
    const qrData = selectedBatchForQR.batchId;
    
    // Use QR code API to generate image
    const qrCodeUrl = `https://api.qrserver.com/v1/create-qr-code/?size=400x400&data=${encodeURIComponent(qrData)}`;
    
    // Download the QR code
    fetch(qrCodeUrl)
      .then(response => response.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = `QR-${selectedBatchForQR.batchId}.png`;
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
      })
      .catch(err => console.error('QR download failed:', err));
  };

  // QR Modal Component
  const QRModal = () => {
    if (!showQRModal || !selectedBatchForQR) return null;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-xl max-w-md w-full p-6 relative">
          <button
            onClick={() => setShowQRModal(false)}
            className="absolute top-4 right-4 p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-teal-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <QrCode className="h-8 w-8 text-teal-600" />
            </div>
            
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Batch QR Code</h2>
            <p className="text-gray-600 mb-6">Scan to verify product authenticity</p>
            
            {/* QR Code Display */}
            <div className="bg-white border-4 border-teal-500 rounded-lg p-6 mb-6">
              <div className="bg-white p-4 rounded flex items-center justify-center">
                <img 
                  src={`https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=${encodeURIComponent(selectedBatchForQR.batchId)}`}
                  alt={`QR Code for ${selectedBatchForQR.batchId}`}
                  className="w-64 h-64"
                />
              </div>
            </div>
            
            {/* Batch Info */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6 text-left">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Batch ID:</span>
                  <span className="font-semibold text-gray-900">{selectedBatchForQR.batchId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Product:</span>
                  <span className="font-semibold text-gray-900">{selectedBatchForQR.productType}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Stage:</span>
                  <span className="font-semibold text-gray-900">{selectedBatchForQR.currentStage}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Location:</span>
                  <span className="font-semibold text-gray-900">{selectedBatchForQR.currentLocation}</span>
                </div>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex space-x-3">
              <button
                onClick={downloadQRCode}
                className="flex-1 bg-teal-600 hover:bg-teal-700 text-white py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Download</span>
              </button>
              <button
                onClick={() => setShowQRModal(false)}
                className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-700 py-3 px-4 rounded-lg font-medium transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="min-h-screen ocean-bg flex items-center justify-center">
        <div className="glass-card p-8 text-center">
          <div className="animate-spin h-8 w-8 border-4 border-teal-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-teal-700">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  const getDashboardTitle = () => {
    switch (user.role) {
      case UserRole.ADMIN: return 'Admin Dashboard';
      case UserRole.PRODUCER: return 'Producer Dashboard';
      case UserRole.AGGREGATOR: return 'Aggregator Dashboard';
      case UserRole.RETAILER: return 'Retailer Dashboard';
      case UserRole.TRANSPORTER: return 'Transporter Dashboard';
      case UserRole.CONSUMER: return 'Consumer Overview';
      default: return 'Dashboard';
    }
  };

  // Admin Dashboard
  if (user.role === UserRole.ADMIN) {
    return (
      <DashboardLayout title="Admin Dashboard">
        <div className="space-y-3 pb-8">
          {/* Compact Header */}
          <div className="glass-card-dark text-white rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-lg font-bold">{getDashboardTitle()}</h1>
                <p className="text-teal-200 text-xs">System administration and monitoring</p>
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-2 px-2 py-1 bg-teal-600 bg-opacity-50 rounded text-xs">
                  <div className="w-1.5 h-1.5 bg-green-400 rounded-full"></div>
                  <span>Active</span>
                </div>
              </div>
            </div>
          </div>

          {/* Ultra Compact Stats Grid */}
          <div className="grid grid-cols-4 gap-2">
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Package className="h-4 w-4 text-teal-600" />
                <div>
                  <p className="text-xs text-gray-600">Total</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Truck className="h-4 w-4 text-blue-600" />
                <div>
                  <p className="text-xs text-gray-600">Transit</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'In Transit').length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <MapPin className="h-4 w-4 text-purple-600" />
                <div>
                  <p className="text-xs text-gray-600">Store</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'At Retailer').length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <AlertTriangle className="h-4 w-4 text-orange-600" />
                <div>
                  <p className="text-xs text-gray-600">Alerts</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.reduce((acc, b) => acc + (b.alerts?.length || 0), 0) : 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Compact System Status & Activity */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="glass-card p-3">
              <h3 className="text-sm font-semibold text-teal-900 mb-2">System Status</h3>
              <AdminWallet />
            </div>

            <div className="glass-card p-3">
              <h3 className="text-sm font-semibold text-teal-900 mb-2">Recent Activity</h3>
              <div className="space-y-1">
                {recentActivity.length > 0 ? recentActivity.slice(0, 4).map((activity) => {
                  const StageIcon = activity.stageIcon || Package;
                  return (
                    <div key={activity.id} className={`p-2 rounded text-xs ${activity.stageBackground || getStageBackground(activity.action)}`}>
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center space-x-2">
                          <StageIcon className={`w-4 h-4 ${activity.stageColor || 'text-gray-600'}`} />
                          <div>
                            <p className="font-semibold text-teal-900">{activity.batchId}</p>
                            <p className="text-gray-500 text-xs">{activity.productType}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <span className="text-gray-500 text-xs">
                            {new Date(activity.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className={`font-bold text-xs px-2 py-1 rounded ${activity.stageColor || 'text-gray-600'} bg-white bg-opacity-70`}>
                            {activity.action}
                          </span>
                          <span className="text-gray-400">→</span>
                          <span className="text-gray-700 font-medium text-xs">{activity.location}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          {activity.stageProgress && (
                            <div className="w-12 bg-gray-200 rounded-full h-1">
                              <div 
                                className="bg-teal-600 h-1 rounded-full transition-all duration-300" 
                                style={{ width: `${activity.stageProgress}%` }}
                              ></div>
                            </div>
                          )}
                          <span className="text-gray-400 text-xs">{activity.updatedBy}</span>
                        </div>
                      </div>
                    </div>
                  );
                }) : (
                  <div className="text-center py-2">
                    <Clock className="h-4 w-4 text-gray-400 mx-auto mb-1" />
                    <p className="text-xs text-gray-500">No activity</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Compact Action Buttons */}
          <div className="flex space-x-2">
            <button 
              onClick={() => router.push('/consumer-audit')}
              className="btn-primary flex items-center space-x-1 flex-1 text-sm py-2"
            >
              <QrCode className="h-3 w-3" />
              <span>QR Scanner</span>
            </button>
            <button 
              onClick={loadDashboardData}
              className="btn-secondary flex items-center space-x-1 text-sm py-2 px-3"
            >
              <RefreshCw className="h-3 w-3" />
              <span>Refresh</span>
            </button>
          </div>

          {/* Compact Batch List */}
          {Array.isArray(batches) && batches.length > 0 && (
            <div className="glass-card p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-teal-900">System Batches ({batches.length})</h3>
                <span className="text-xs text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">Admin</span>
              </div>
              
              <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {batches.map((batch) => (
                  <div key={batch.batchId} className="flex items-center justify-between p-2 border border-gray-200 rounded bg-gray-50 bg-opacity-50">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${batch.isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                      <div>
                        <h4 className="font-semibold text-teal-900 text-xs">{batch.batchId}</h4>
                        <p className="text-xs text-gray-600">{batch.productType} • {getDisplayStage(batch.currentStage, user?.role)}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div>
                        <p className="text-xs text-gray-500">{batch.currentLocation}</p>
                        {batch.locationHistory && batch.locationHistory[0] && (
                          batch.locationHistory[0].transactionHash.startsWith('0x') && batch.locationHistory[0].transactionHash.length === 66 ? (
                            <a
                              href={`https://explorer-mezame.shardeum.org/tx/${batch.locationHistory[0].transactionHash}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 text-xs flex items-center space-x-1"
                              title="View transaction on Shardeum Explorer"
                            >
                              <ExternalLink className="h-2 w-2" />
                              <span>TX</span>
                            </a>
                          ) : (
                            <span 
                              className="text-gray-400 text-xs flex items-center space-x-1 cursor-not-allowed"
                              title="Demo transaction - not available on explorer"
                            >
                              <ExternalLink className="h-2 w-2" />
                              <span>Demo</span>
                            </span>
                          )
                        )}
                      </div>
                      <button
                        onClick={() => generateQRCode(batch)}
                        className="p-1 hover:bg-teal-100 rounded transition-colors"
                        title="Generate QR Code"
                      >
                        <QrCode className="h-4 w-4 text-teal-600" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <QRModal />
      </DashboardLayout>
    );
  }

  // Aggregator/Producer Dashboard (Creates Batches)
  if (user.role === UserRole.AGGREGATOR || user.role === UserRole.PRODUCER) {
    const roleTitle = user.role === UserRole.PRODUCER ? 'Producer Dashboard' : 'Aggregator Dashboard';
    return (
      <DashboardLayout title={roleTitle}>
        <div className="space-y-3 pb-8">
          {/* Compact Header */}
          <div className="glass-card-dark text-white rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-lg font-bold">{getDashboardTitle()}</h1>
                <p className="text-teal-200 text-xs">Record harvests in DB • Manage blockchain batches</p>
              </div>
              <div className="flex space-x-1">
                <button 
                  onClick={loadDashboardData}
                  className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-2 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <RefreshCw className="h-3 w-3" />
                  <span>Refresh</span>
                </button>
                <button
                  onClick={() => router.push('/marketplace')}
                  className="bg-white bg-opacity-10 hover:bg-opacity-30 text-white px-2 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <ShoppingBag className="h-3 w-3" />
                  <span>Marketplace</span>
                </button>
              </div>
            </div>
          </div>

          {/* Ultra Compact Stats */}
          <div className="grid grid-cols-4 gap-2">
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Package className="h-4 w-4 text-teal-600" />
                <div>
                  <p className="text-xs text-gray-600">Created</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <div>
                  <p className="text-xs text-gray-600">Active</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.isActive).length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Truck className="h-4 w-4 text-blue-600" />
                <div>
                  <p className="text-xs text-gray-600">Transit</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'In Transit').length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <AlertTriangle className="h-4 w-4 text-orange-600" />
                <div>
                  <p className="text-xs text-gray-600">Alerts</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.reduce((acc, b) => acc + (b.alerts?.length || 0), 0) : 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Compact Action Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="glass-card p-3 text-center">
              <ShoppingBag className="h-8 w-8 text-emerald-600 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-teal-900 mb-1">Record Harvest</h3>
              <p className="text-gray-600 text-xs mb-2">Log parent batch to DB</p>
              <button 
                onClick={() => router.push('/marketplace')}
                className="btn-primary w-full text-xs py-1.5"
              >
                Record
              </button>
            </div>

            <div className="glass-card p-3 text-center">
              <QrCode className="h-8 w-8 text-blue-600 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-teal-900 mb-1">QR Scanner</h3>
              <p className="text-gray-600 text-xs mb-2">Verify existing batches</p>
              <button 
                onClick={() => router.push('/consumer-audit')}
                className="btn-secondary w-full text-xs py-1.5"
              >
                Scan
              </button>
            </div>
          </div>

          {/* Compact Created Batches */}
          {Array.isArray(batches) && batches.length > 0 ? (
            <div className="glass-card p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-teal-900">Created Batches ({batches.length})</h3>
                <span className="text-xs text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">Latest First</span>
              </div>
              
              <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {batches.map((batch) => (
                  <div key={batch.batchId} className="border border-gray-200 rounded p-2 bg-gray-50 bg-opacity-50">
                    <div className="flex items-center justify-between mb-1">
                      <div>
                        <h4 className="font-semibold text-teal-900 text-xs">{batch.batchId}</h4>
                        <p className="text-xs text-gray-600">{batch.productType}</p>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${getStageColor(batch.currentStage)} bg-white bg-opacity-70`}>
                          {batch.currentStage}
                        </span>
                        <p className="text-xs text-gray-500 mt-0.5">{batch.currentLocation}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>{new Date(batch.created).toLocaleDateString()}</span>
                      <div className="flex items-center space-x-2">
                        {batch.locationHistory && batch.locationHistory[0] && (
                          batch.locationHistory[0].transactionHash.startsWith('0x') && batch.locationHistory[0].transactionHash.length === 66 ? (
                            <a
                              href={`https://explorer-mezame.shardeum.org/tx/${batch.locationHistory[0].transactionHash}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 flex items-center space-x-1"
                              title="View transaction on Shardeum Explorer"
                            >
                              <ExternalLink className="h-2 w-2" />
                              <span>TX</span>
                            </a>
                          ) : (
                            <span 
                              className="text-gray-400 flex items-center space-x-1 cursor-not-allowed"
                              title="Demo transaction - not available on explorer"
                            >
                              <ExternalLink className="h-2 w-2" />
                              <span>Demo</span>
                            </span>
                          )
                        )}
                        <button
                          onClick={() => generateQRCode(batch)}
                          className="p-1 hover:bg-teal-100 rounded transition-colors"
                          title="Generate QR Code"
                        >
                          <QrCode className="h-3 w-3 text-teal-600" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="glass-card text-center py-6">
              <Package className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-gray-900 mb-1">No Batches Recorded</h3>
              <p className="text-gray-600 text-xs">Waiting for the first batch update.</p>
            </div>
          )}
        </div>
        <QRModal />
      </DashboardLayout>
    );
  }

  // Retailer Dashboard
  if (user.role === UserRole.RETAILER) {
    return (
      <DashboardLayout title="Retailer Dashboard">
        <div className="space-y-3 pb-8">
          {/* Compact Header */}
          <div className="glass-card-dark text-white rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-lg font-bold">{getDashboardTitle()}</h1>
                <p className="text-teal-200 text-xs">View batch details and workflow stages</p>
              </div>
              <div className="flex space-x-1">
                <button 
                  onClick={loadDashboardData}
                  className="bg-teal-600 hover:bg-teal-700 text-white px-2 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <RefreshCw className="h-3 w-3" />
                  <span>Refresh</span>
                </button>
                <button 
                  onClick={() => router.push('/consumer-audit')}
                  className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-2 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <Eye className="h-3 w-3" />
                  <span>Scan</span>
                </button>
                <button
                  onClick={() => router.push('/marketplace')}
                  className="bg-white bg-opacity-10 hover:bg-opacity-30 text-white px-2 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <ShoppingBag className="h-3 w-3" />
                  <span>Marketplace</span>
                </button>
              </div>
            </div>
          </div>

          {/* Ultra Compact Stats */}
          <div className="grid grid-cols-3 gap-2">
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Package className="h-4 w-4 text-teal-600" />
                <div>
                  <p className="text-xs text-gray-600">Total</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <Truck className="h-4 w-4 text-blue-600" />
                <div>
                  <p className="text-xs text-gray-600">Transit</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'In Transit').length : 0}
                  </p>
                </div>
              </div>
            </div>
            <div className="glass-card p-2">
              <div className="flex items-center space-x-1">
                <MapPin className="h-4 w-4 text-purple-600" />
                <div>
                  <p className="text-xs text-gray-600">Store</p>
                  <p className="text-sm font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'At Retailer').length : 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Ultra Compact Batch List */}
          {Array.isArray(batches) && batches.length > 0 ? (
            <div className="glass-card p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-teal-900">Your At Retailer Batches ({(() => {
                  const filtered = batches.filter(b => 
                    b.currentStage === 'At Retailer'
                  );
                  return filtered.length;
                })()})</h3>
                <span className="text-xs text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">Ready for sale</span>
              </div>
              
              <div className="space-y-1.5">
                {(() => {
                  const filtered = batches.filter(b => 
                    b.currentStage === 'At Retailer'
                  );
                  return filtered.length > 0 ? filtered.map((batch) => (
                  <div key={batch.batchId} className="border border-gray-200 rounded p-2 bg-gray-50 bg-opacity-50">
                    {/* Ultra Compact Batch Header */}
                    <div className="flex items-center justify-between mb-1">
                      <div>
                        <h4 className="font-semibold text-teal-900 text-xs">{batch.batchId}</h4>
                        <p className="text-xs text-gray-600">{batch.productType}</p>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${getStageColor(batch.currentStage)} bg-white bg-opacity-70`}>
                          {getDisplayStage(batch.currentStage, user?.role)}
                        </span>
                        <p className="text-xs text-gray-500 mt-0.5">{batch.currentLocation}</p>
                      </div>
                    </div>

                    {/* Ultra Compact Timeline */}
                    <div className="border-t border-gray-200 pt-1 mt-1">
                      <div className="grid grid-cols-2 gap-1">
                        {batch.locationHistory && batch.locationHistory.length > 0 ? (
                          batch.locationHistory.slice(0, 2).map((history, index) => (
                            <div key={index} className="flex items-center space-x-1 text-xs">
                              <div className={`w-1.5 h-1.5 rounded-full ${
                                index === 0 ? 'bg-green-500' : 'bg-gray-300'
                              }`}></div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between">
                                  <span className="font-medium text-gray-900 truncate text-xs">{history.stage}</span>
                                  <span className="text-gray-500 ml-1 text-xs">
                                    {new Date(history.timestamp).toLocaleDateString()}
                                  </span>
                                </div>
                                <p className="text-gray-600 truncate text-xs">{history.location}</p>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-xs text-gray-500 col-span-2">No history</p>
                        )}
                      </div>
                    </div>

                    {/* Compact Alerts */}
                    {batch.alerts && batch.alerts.length > 0 && (
                      <div className="border-t border-gray-200 pt-1 mt-1">
                        <div className="flex items-center space-x-1">
                          <AlertTriangle className="h-3 w-3 text-orange-600" />
                          <span className="text-xs text-orange-800">
                            {batch.alerts.length} alert{batch.alerts.length > 1 ? 's' : ''}
                          </span>
                        </div>
                      </div>
                    )}
                    
                    {/* QR Code Button */}
                    <div className="border-t border-gray-200 pt-1 mt-1">
                      <button
                        onClick={() => generateQRCode(batch)}
                        className="w-full flex items-center justify-center space-x-2 px-2 py-1 text-xs text-teal-600 hover:bg-teal-50 rounded transition-colors"
                      >
                        <QrCode className="h-3 w-3" />
                        <span>Generate QR Code</span>
                      </button>
                    </div>
                  </div>
                  )) : (
                    <div className="text-center py-4 text-gray-500 text-sm">
                      <p>No batches at retailer stage scanned by you.</p>
                      <p className="text-xs mt-1">Scan a QR code to add batches to your dashboard.</p>
                    </div>
                  );
                })()}
              </div>
            </div>
          ) : (
            <div className="glass-card text-center py-6">
              <Package className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-gray-900 mb-1">No Batches Available</h3>
              <p className="text-gray-600 text-xs mb-3">No batches found to display workflow information.</p>
              <button 
                onClick={loadDashboardData}
                className="btn-secondary flex items-center space-x-1 mx-auto text-xs py-1.5 px-3"
              >
                <RefreshCw className="h-3 w-3" />
                <span>Refresh Data</span>
              </button>
            </div>
          )}
        </div>
        <QRModal />
      </DashboardLayout>
    );
  }

  // Transporter Dashboard
  if (user.role === UserRole.TRANSPORTER) {
    return (
      <DashboardLayout title="Transporter Dashboard">
        <div className="space-y-6 pb-24">
          {/* Welcome Banner */}
          <div className="welcome-banner">
            <h1 className="text-2xl font-bold">{getDashboardTitle()}</h1>
            <p className="text-teal-700 mt-2">Manage batch transportation</p>
          </div>

          {/* Stats Card */}
          <div className="metric-card">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-teal-900">Batches in Transit</h3>
                <div className="flex items-center space-x-2 mt-2">
                  <Truck className="h-6 w-6 text-emerald-600" />
                  <span className="text-3xl font-bold text-teal-900">
                    {Array.isArray(batches) ? batches.filter(b => b.currentStage === 'In Transit').length : 0}
                  </span>
                </div>
                <p className="text-sm text-gray-500">{new Date().toLocaleDateString()}</p>
              </div>
              <div className="text-emerald-600">
                <Package className="h-8 w-8" />
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="flex space-x-4">
            <button 
              onClick={() => router.push('/professional')}
              className="btn-primary flex items-center space-x-2 w-full justify-center"
            >
              <span>Manage Batches</span>
            </button>
          </div>

          {/* Batch List */}
          {(() => {
            const userBatches = Array.isArray(batches) 
              ? batches.filter(b => 
                  b.currentStage === 'In Transit'
                )
              : [];
            
            return userBatches.length > 0 && (
              <div className="metric-card">
                <h3 className="text-lg font-semibold text-teal-900 mb-4">Your In Transit Batches ({userBatches.length})</h3>
                <div className="space-y-3">
                  {userBatches.map((batch) => (
                    <div key={batch.batchId} className="p-3 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <div className={`w-3 h-3 rounded-full ${batch.isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                          <h4 className="font-bold text-teal-900">{batch.batchId}</h4>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
                          {getDisplayStage(batch.currentStage, user?.role)}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <p className="text-gray-500">Product</p>
                          <p className="text-gray-900 font-medium">{batch.productType}</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Location</p>
                          <p className="text-gray-900 font-medium">{batch.currentLocation}</p>
                        </div>
                      </div>
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <button
                          onClick={() => generateQRCode(batch)}
                          className="w-full flex items-center justify-center space-x-2 px-3 py-1.5 text-sm text-teal-600 hover:bg-teal-50 rounded transition-colors"
                        >
                          <QrCode className="h-4 w-4" />
                          <span>Generate QR Code</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* Professional Tools Link */}
          <div className="metric-card text-center py-8">
            <Truck className="h-12 w-12 text-emerald-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-teal-900 mb-2">Transport Management</h3>
            <p className="text-gray-600 mb-4">Update batch locations and report transport status.</p>
            <button 
              onClick={() => router.push('/professional')}
              className="btn-primary"
            >
              Open Transport Tools
            </button>
          </div>
        </div>
        <QRModal />
      </DashboardLayout>
    );
  }

  // Consumer Dashboard
  return (
    <DashboardLayout title="Consumer Overview">
      <div className="space-y-6 pb-24">
        {/* Consumer QR Scanning */}
        <div className="metric-card text-center py-12">
          <Camera className="h-16 w-16 text-blue-600 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-teal-900 mb-2">Consumer Access</h2>
          <p className="text-gray-600 mb-6">
            Scan QR codes with your device camera to view product information and verify authenticity.
          </p>
          <button
            onClick={() => router.push('/consumer-audit')}
            className="btn-primary inline-flex items-center space-x-2"
          >
            <QrCode className="h-5 w-5" />
            <span>Start QR Scanning</span>
          </button>
        </div>

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="metric-card">
            <h3 className="text-lg font-semibold text-teal-900 mb-4">How to Scan</h3>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>Find the QR code on the product packaging</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>Click "Start QR Scanning" above</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>Point your camera at the QR code</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>View complete product journey</span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <h3 className="text-lg font-semibold text-teal-900 mb-4">What You'll See</h3>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Product origin and type</span>
              </div>
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Complete supply chain timeline</span>
              </div>
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Quality and freshness status</span>
              </div>
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Blockchain verification</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <QRModal />
    </DashboardLayout>
  );
};

export default DashboardPage;