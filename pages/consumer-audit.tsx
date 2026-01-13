import { NextPage } from 'next';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Camera, Package, Search, CheckCircle, Clock, MapPin, AlertTriangle, AlertCircle, Wifi, Radio, Upload, Leaf, ArrowLeft } from 'lucide-react';
import { getBackendService, BatchDetails, SensorType, SensorInfo } from '../lib/services/backendService';
import { getAuthService, UserRole } from '../lib/services/authService';
import QRScanner from '../components/common/QRScanner';

const ConsumerAuditPage: NextPage = () => {
  const router = useRouter();
  const [batchId, setBatchId] = useState<string>('');
  const [batchData, setBatchData] = useState<BatchDetails | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [showScanner, setShowScanner] = useState(false);
  
  // Sensor selection
  const [availableSensors, setAvailableSensors] = useState<SensorInfo[]>([]);
  const [selectedSensorId, setSelectedSensorId] = useState<string>('');
  const [showSensorSelection, setShowSensorSelection] = useState(false);
  const [sensorId, setSensorId] = useState('');
  const [sensorLinkStatus, setSensorLinkStatus] = useState('');
  const [sensorLinkSuccess, setSensorLinkSuccess] = useState<boolean | null>(null);
  const [isLinkingSensor, setIsLinkingSensor] = useState(false);
  const [lastQrPayload, setLastQrPayload] = useState('');
  
  // Mandatory sample testing
  const [sampleImage, setSampleImage] = useState<File | null>(null);
  const [sampleImagePreview, setSampleImagePreview] = useState<string>('');
  const [showSampleTestingModal, setShowSampleTestingModal] = useState(false);

  const backendService = getBackendService();
  const authService = getAuthService();
  const user = authService.getUser();
  const canLinkSensors = !!user && (user.role === UserRole.RETAILER || user.role === UserRole.TRANSPORTER);
  const isRetailer = user?.role === UserRole.RETAILER;
  const locationType: SensorType = user?.role === UserRole.TRANSPORTER ? SensorType.TRANSPORTER : SensorType.RETAILER;
  const resolveBatchIdFromPayload = (payload: string) => {
    try {
      const parsed = JSON.parse(payload);
      return typeof parsed.batchId === 'string' ? parsed.batchId : payload;
    } catch {
      return payload;
    }
  };

  const derivePayloadForBatch = (id: string) => {
    if (lastQrPayload) {
      const existing = resolveBatchIdFromPayload(lastQrPayload);
      if (existing === id) {
        return lastQrPayload;
      }
    }
    return JSON.stringify({ batchId: id });
  };

  const loadAvailableSensors = async () => {
    if (!canLinkSensors) return;
    
    try {
      const response = await backendService.getAvailableSensors(locationType);
      if (response.success && response.data) {
        const unlinkedSensors = response.data.sensors.filter(s => !s.isLinked);
        setAvailableSensors(unlinkedSensors);
      }
    } catch (err) {
      console.error('Failed to load sensors:', err);
    }
  };

  const handleSampleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file');
        return;
      }
      setSampleImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setSampleImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const convertImageToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data:image/xxx;base64, prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleSensorSelection = async (selectedSensor: string) => {
    if (!batchId.trim()) {
      setSensorLinkStatus('Please scan or enter a batch ID first');
      setSensorLinkSuccess(false);
      return;
    }

    // Set the selected sensor ID
    setSelectedSensorId(selectedSensor);
    setShowSensorSelection(false);
    
    // Automatically trigger the sensor linking with the selected sensor
    const payload = derivePayloadForBatch(batchId.trim());
    await linkSensorToBatch(batchId.trim(), payload, selectedSensor);
  };

  const linkSensorToBatch = async (targetBatchId: string, qrPayload: string, sensorIdOverride?: string) => {
    if (!canLinkSensors) {
      return;
    }
    
    const effectiveSensorId = sensorIdOverride || sensorId.trim();
    
    if (!effectiveSensorId) {
      setSensorLinkStatus('Provide your sensor ID to link scans with live sensors.');
      setSensorLinkSuccess(false);
      return;
    }

    // Check for mandatory sample testing (only for retailers)
    if (isRetailer && !sampleImage) {
      setShowSampleTestingModal(true);
      setSensorLinkStatus('Sample testing is mandatory for retailers. Please upload a product image.');
      setSensorLinkSuccess(false);
      return;
    }

    setIsLinkingSensor(true);
    setSensorLinkStatus(isRetailer ? 'Processing sample test and linking sensor...' : 'Linking sensor...');
    setSensorLinkSuccess(null);

    try {
      // Convert image to base64 (only if sample image provided)
      const sampleImageBase64 = sampleImage ? await convertImageToBase64(sampleImage) : undefined;
      
      const response = await backendService.linkSensorToBatch({
        batchId: targetBatchId,
        sensorId: effectiveSensorId,
        locationType,
        qrPayload,
        sampleImageBase64,
      });

      if (response.success) {
        const message = isRetailer 
          ? `Batch ${targetBatchId} added to your dashboard! Sensor ${sensorId.trim()} linked successfully. Sample test completed.`
          : `Batch ${targetBatchId} added to your dashboard! Sensor ${sensorId.trim()} linked successfully.`;
        setSensorLinkStatus(message);
        setSensorLinkSuccess(true);
        // Clear sensor selection UI and sample image
        setAvailableSensors([]);
        setSampleImage(null);
        setSampleImagePreview('');
      } else {
        setSensorLinkStatus(response.error || 'Unable to link sensor.');
        setSensorLinkSuccess(false);
      }
    } catch (err) {
      setSensorLinkStatus(err instanceof Error ? err.message : 'Unable to link sensor.');
      setSensorLinkSuccess(false);
    } finally {
      setIsLinkingSensor(false);
    }
  };

  const handleManualSensorLink = () => {
    if (!canLinkSensors) {
      return;
    }
    if (!batchId.trim()) {
      setSensorLinkStatus('Lookup a batch or scan a QR before linking sensors.');
      setSensorLinkSuccess(false);
      return;
    }
    const payload = derivePayloadForBatch(batchId.trim());
    void linkSensorToBatch(batchId.trim(), payload);
  };
  const handleSearch = async () => {
    if (!batchId.trim()) {
      setError('Please enter a batch ID');
      return;
    }

    setIsLoading(true);
    setError('');
    setBatchData(null);

    try {
      const result = await backendService.getBatchDetails(batchId.trim());
      if (result.success && result.data) {
        setBatchData(result.data);
      } else {
        setError(result.error || 'Batch not found');
      }
    } catch (err) {
      setError('Failed to fetch batch details');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQRScan = async (qrData: string) => {
    setShowScanner(false);
    const resolvedBatchId = resolveBatchIdFromPayload(qrData);
    setBatchId(resolvedBatchId);
    setLastQrPayload(qrData);
    
    // Auto-search after QR scan
    setIsLoading(true);
    setError('');
    setBatchData(null);

    try {
      const result = await backendService.getBatchDetails(resolvedBatchId);
      if (result.success && result.data) {
        setBatchData(result.data);
        if (canLinkSensors) {
          void linkSensorToBatch(resolvedBatchId, qrData);
        }
      } else {
        setError(result.error || 'Batch not found');
      }
    } catch (err) {
      setError('Failed to fetch batch details');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'Created':
      case 'Harvested':
        return <Package className="h-4 w-4" />;
      case 'In Transit':
        return <Clock className="h-4 w-4" />;
      case 'At Retailer':
      case 'Selling':
        return <MapPin className="h-4 w-4" />;
      default:
        return <CheckCircle className="h-4 w-4" />;
    }
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'Created':
        return 'bg-gray-100 text-gray-700';
      case 'Harvested':
        return 'bg-green-100 text-green-700';
      case 'In Transit':
        return 'bg-blue-100 text-blue-700';
      case 'At Retailer':
        return 'bg-purple-100 text-purple-700';
      case 'Selling':
        return 'bg-emerald-100 text-emerald-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="header-bar">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => router.back()}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Go back"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </button>
          <div className="w-8 h-8 bg-freshchain-primary rounded-lg flex items-center justify-center">
            <Camera className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Consumer Audit</h1>
            <p className="text-sm text-gray-500 hidden sm:block">Verify product authenticity</p>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl p-8 mb-8">
          <div className="text-center">
            <Camera className="h-16 w-16 mx-auto mb-4 opacity-90" />
            <h1 className="text-3xl font-bold mb-4">Product Verification</h1>
            <p className="text-emerald-100 text-lg mb-6">
              Scan QR codes or enter batch IDs to verify product authenticity and view supply chain journey
            </p>
            <div className="inline-flex items-center px-4 py-2 bg-white bg-opacity-20 rounded-lg">
              <CheckCircle className="h-5 w-5 mr-2" />
              <span className="text-sm font-medium">No account required • Instant verification</span>
            </div>
          </div>
        </div>

        {/* Search Section */}
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Verify Product</h2>
          
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="flex-1">
              <input
                type="text"
                value={batchId}
                onChange={(e) => setBatchId(e.target.value)}
                placeholder="Enter batch ID (e.g., APPLE-001)"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={isLoading}
              className="px-6 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors disabled:opacity-50 flex items-center justify-center"
            >
              {isLoading ? (
                <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
              ) : (
                <>
                  <Search className="h-4 w-4 mr-2" />
                  Search
                </>
              )}
            </button>
          </div>

          <div className="text-center">
            <span className="text-gray-500 text-sm">or</span>
            <button
              onClick={() => setShowScanner(true)}
              className="ml-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center mx-auto mt-2"
            >
              <Camera className="h-4 w-4 mr-2" />
              Scan QR Code
            </button>
          </div>
        </div>

        {canLinkSensors && (
          <div className="card mb-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Sensor Linking (Retail & Transport)</h2>
            <p className="text-sm text-gray-600 mb-4">
              Provide your assigned sensor ID and FreshChain will automatically bind scanned batches to your device. Manual linking remains available below.
            </p>
            
            {/* Mandatory Sample Testing - Only for Retailers */}
            {isRetailer && (
              <div className="mb-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Leaf className="h-5 w-5 text-amber-600" />
                  <h3 className="font-semibold text-amber-900">Mandatory Sample Testing</h3>
                </div>
                <p className="text-sm text-amber-800 mb-3">
                  Upload a clear image of the product for AI freshness analysis before linking the sensor.
                </p>
                <div className="flex flex-col md:flex-row gap-4 items-start">
                  <div className="flex-1">
                    <label className="block">
                      <div className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-amber-500 transition-colors ${
                        sampleImage ? 'border-amber-500 bg-amber-50' : 'border-gray-300 bg-gray-50'
                      }`}>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleSampleImageUpload}
                          className="hidden"
                        />
                        <Upload className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                        <p className="text-sm text-gray-600">
                          {sampleImage ? `Selected: ${sampleImage.name}` : 'Click to upload product image'}
                        </p>
                      </div>
                    </label>
                  </div>
                  {sampleImagePreview && (
                    <div className="w-32 h-32">
                      <img
                        src={sampleImagePreview}
                        alt="Sample preview"
                        className="w-full h-full object-cover rounded-lg border-2 border-amber-500"
                      />
                    </div>
                  )}
                </div>
                {!sampleImage && (
                  <p className="text-xs text-amber-700 mt-2">
                    ⚠️ Sample testing is required before sensor linking
                  </p>
                )}
              </div>
            )}
            
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="text"
                value={sensorId}
                onChange={(e) => {
                  setSensorId(e.target.value);
                  setSensorLinkStatus('');
                  setSensorLinkSuccess(null);
                }}
                placeholder="Sensor hardware ID (e.g., RETAILER-001)"
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              />
              <button
                onClick={handleManualSensorLink}
                disabled={isLinkingSensor}
                className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50"
              >
                {isLinkingSensor ? 'Linking...' : 'Link Current Batch'}
              </button>
            </div>
            {sensorLinkStatus && (
              <p className={`mt-3 text-sm ${sensorLinkSuccess ? 'text-emerald-600' : 'text-amber-600'}`}>
                {sensorLinkStatus}
              </p>
            )}
            <p className="text-xs text-gray-500 mt-2">
              Tip: scan the QR at your store or truck after entering the sensor ID and FreshChain will connect automatically.
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="card bg-red-50 border-red-200 mb-8">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              <span className="text-red-800 font-medium">Error</span>
            </div>
            <p className="text-red-700 mt-1">{error}</p>
          </div>
        )}

        {/* Batch Details */}
        {batchData && (
          <div className="space-y-6">
            {/* Sensor Linking Section - Only for Retailers/Transporters */}
            {canLinkSensors && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Wifi className="h-5 w-5 mr-2 text-blue-600" />
                  Link IoT Sensor
                </h3>

                {/* Sensor Selection UI */}
                {showSensorSelection && availableSensors.length > 0 && (
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 mb-3">Select an available sensor to monitor this batch:</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {availableSensors.map((sensor) => (
                        <button
                          key={sensor.sensorId}
                          onClick={() => handleSensorSelection(sensor.sensorId)}
                          disabled={isLinkingSensor}
                          className="p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all text-left disabled:opacity-50"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center">
                              <Radio className="h-5 w-5 text-blue-600 mr-2" />
                              <div>
                                <p className="font-semibold text-gray-900">{sensor.sensorId}</p>
                                {sensor.label && (
                                  <p className="text-xs text-gray-500">{sensor.label}</p>
                                )}
                                {sensor.vehicleOrStoreId && (
                                  <p className="text-xs text-gray-500">{sensor.vehicleOrStoreId}</p>
                                )}
                              </div>
                            </div>
                            <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">
                              Available
                            </span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {showSensorSelection && availableSensors.length === 0 && (
                  <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-yellow-800 text-sm">
                      No available sensors found. Register sensors via the Sensors dashboard first.
                    </p>
                  </div>
                )}

                {!showSensorSelection && (
                  <button
                    onClick={async () => {
                      await loadAvailableSensors();
                      setShowSensorSelection(true);
                    }}
                    className="btn-secondary"
                    disabled={isLinkingSensor}
                  >
                    <Wifi className="h-4 w-4 mr-2" />
                    Select Sensor to Monitor
                  </button>
                )}

                {sensorLinkStatus && (
                  <div className={`mt-3 p-3 rounded-lg flex items-center justify-between ${
                    sensorLinkSuccess === true
                      ? 'bg-green-50 text-green-800 border border-green-200'
                      : sensorLinkSuccess === false
                      ? 'bg-red-50 text-red-800 border border-red-200'
                      : 'bg-blue-50 text-blue-800 border border-blue-200'
                  }`}>
                    <div className="flex items-center">
                      {sensorLinkSuccess === true && <CheckCircle className="h-5 w-5 mr-2 text-green-600" />}
                      {sensorLinkSuccess === false && <AlertCircle className="h-5 w-5 mr-2 text-red-600" />}
                      <span>{sensorLinkStatus}</span>
                    </div>
                    {sensorLinkSuccess === true && (
                      <button
                        onClick={() => router.push('/dashboard')}
                        className="ml-4 px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2 whitespace-nowrap"
                      >
                        <Package className="h-4 w-4" />
                        View Dashboard
                      </button>
                    )}
                  </div>
                )}

                {selectedSensorId && sensorLinkSuccess && (
                  <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center">
                    <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                    <span className="text-green-800 font-medium">
                      Sensor {selectedSensorId} is actively monitoring this batch
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Product Info */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900">Product Information</h3>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-green-700 font-medium">Verified</span>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Batch ID</label>
                  <div className="text-lg font-mono bg-gray-50 px-3 py-2 rounded border">
                    {batchData.batchId}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Product Type</label>
                  <div className="text-lg bg-gray-50 px-3 py-2 rounded border">
                    {batchData.productType}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Created</label>
                  <div className="text-lg bg-gray-50 px-3 py-2 rounded border">
                    {formatDate(batchData.created)}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Current Status</label>
                  <div className={`inline-flex items-center space-x-2 px-3 py-2 rounded-lg ${getStageColor(batchData.currentStage)}`}>
                    {getStageIcon(batchData.currentStage)}
                    <span className="font-medium">{batchData.currentStage}</span>
                  </div>
                </div>
              </div>

              {/* Shelf Life & Freshness Stats */}
              {(batchData.ageInDays !== undefined || batchData.freshnessScore !== undefined) && (
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-3 flex items-center">
                    <Clock className="h-5 w-5 mr-2" />
                    Time & Freshness Statistics
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {(batchData.ageInDays !== undefined || batchData.ageInHours !== undefined) && (
                      <div>
                        <label className="block text-xs font-medium text-blue-700 mb-1">Product Age</label>
                        <div className="text-2xl font-bold text-blue-900">
                          {batchData.ageInDays !== undefined && batchData.ageInDays < 1 && batchData.ageInHours !== undefined
                            ? `${batchData.ageInHours.toFixed(1)} hours`
                            : batchData.ageInDays !== undefined
                            ? `${batchData.ageInDays.toFixed(1)} days`
                            : 'N/A'}
                        </div>
                      </div>
                    )}
                    {batchData.freshnessScore !== undefined && (
                      <div>
                        <label className="block text-xs font-medium text-blue-700 mb-1">Freshness Score</label>
                        <div className="text-2xl font-bold text-blue-900">
                          {(batchData.freshnessScore * 100).toFixed(0)}%
                        </div>
                        {batchData.freshnessCategory && (
                          <span className={`inline-block mt-1 px-2 py-1 text-xs font-medium rounded ${
                            batchData.freshnessCategory.toLowerCase().includes('fresh')
                              ? 'bg-green-100 text-green-800'
                              : batchData.freshnessCategory.toLowerCase().includes('rotten')
                              ? 'bg-red-100 text-red-800'
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {batchData.freshnessCategory}
                          </span>
                        )}
                      </div>
                    )}
                    {batchData.estimatedShelfLifeDays !== undefined && batchData.estimatedShelfLifeDays !== null && (
                      <div>
                        <label className="block text-xs font-medium text-blue-700 mb-1">Estimated Shelf Life</label>
                        <div className="text-2xl font-bold text-blue-900">
                          {batchData.estimatedShelfLifeDays.toFixed(1)} days
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {batchData.isFinalStage && (
                <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-emerald-600" />
                    <span className="text-emerald-800 font-medium">Final Retail Stage Reached</span>
                  </div>
                  <p className="text-emerald-700 text-sm mt-1">
                    This product has completed its supply chain journey and is ready for sale.
                  </p>
                </div>
              )}
            </div>

            {/* Supply Chain Timeline */}
            <div className="card">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Supply Chain Journey</h3>
              
              {batchData.locationHistory.length > 0 ? (
                <div className="space-y-4">
                  {batchData.locationHistory.map((update, index) => (
                    <div key={index} className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg">
                      <div className={`p-2 rounded-lg ${getStageColor(update.stage)}`}>
                        {getStageIcon(update.stage)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold text-gray-900">{update.stage}</h4>
                          <span className="text-sm text-gray-500">{formatDate(update.timestamp)}</span>
                        </div>
                        <p className="text-gray-600 mt-1">{update.location}</p>
                        {update.updatedBy && (
                          <p className="text-xs text-gray-500 mt-1">Updated by: {update.updatedBy}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No location history available yet</p>
                </div>
              )}
            </div>

            {/* Alerts */}
            {batchData.alerts.length > 0 && (
              <div className="card">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Alerts & Issues</h3>
                <div className="space-y-4">
                  {batchData.alerts.map((alert, index) => (
                    <div key={index} className="flex items-start space-x-4 p-4 bg-orange-50 border border-orange-200 rounded-lg">
                      <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5" />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold text-orange-900">{alert.alertType}</h4>
                          <span className="text-sm text-orange-600">{formatDate(alert.timestamp)}</span>
                        </div>
                        <p className="text-orange-700 mt-1">{alert.encryptedData}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        {!batchData && !isLoading && (
          <div className="card bg-blue-50 border-blue-200">
            <h3 className="text-lg font-semibold text-blue-900 mb-4">How to Use</h3>
            <div className="text-sm text-blue-700 space-y-2">
              <p>• Enter a batch ID in the search field above</p>
              <p>• Click "Scan QR Code" to use your device camera</p>
              <p>• All data is fetched in real-time from the blockchain</p>
              <p>• QR codes are typically found on product packaging</p>
            </div>
          </div>
        )}
      </div>

      {/* QR Scanner Modal */}
      {showScanner && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Scan QR Code</h3>
              <button
                onClick={() => setShowScanner(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
              </button>
            </div>
            <QRScanner
              onScan={handleQRScan}
              onError={(error) => {
                setError(error);
                setShowScanner(false);
              }}
              onClose={() => setShowScanner(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ConsumerAuditPage;