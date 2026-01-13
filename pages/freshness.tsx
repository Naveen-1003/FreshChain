import { NextPage } from 'next';
import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Camera, Upload, Leaf, CheckCircle, XCircle, AlertCircle, Sparkles } from 'lucide-react';
import DashboardLayout from '../components/layout/DashboardLayout';
import { getAuthService } from '../lib/services/authService';
import { getBackendService } from '../lib/services/backendService';

interface FreshnessResult {
  freshnessScore: number;
  freshnessCategory: string;
  confidence: number;
  message: string;
  dominantClass?: string;
  dominantScore?: number;
}

const FreshnessAIPage: NextPage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [batchId, setBatchId] = useState<string>('');
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<FreshnessResult | null>(null);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const router = useRouter();
  const authService = getAuthService();
  const backendService = getBackendService();
  const user = authService.getUser();

  useEffect(() => {
    if (!authService.isAuthenticated()) {
      router.push('/login');
    }
  }, [router]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError('');
      setResult(null);
    }
  };

  const handleScan = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsScanning(true);
    setError('');
    setResult(null);

    try {
      const response = await backendService.scanFreshness(selectedFile, batchId || undefined);
      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || 'Failed to analyze image');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to scan freshness');
    } finally {
      setIsScanning(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setBatchId('');
    setResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getCategoryIcon = (category: string) => {
    const lowerCategory = category.toLowerCase();
    if (lowerCategory.includes('fresh') || lowerCategory.includes('good')) {
      return <CheckCircle className="h-12 w-12 text-green-500" />;
    } else if (lowerCategory.includes('rotten') || lowerCategory.includes('bad')) {
      return <XCircle className="h-12 w-12 text-red-500" />;
    } else {
      return <AlertCircle className="h-12 w-12 text-yellow-500" />;
    }
  };

  const getCategoryColor = (category: string) => {
    const lowerCategory = category.toLowerCase();
    if (lowerCategory.includes('fresh') || lowerCategory.includes('good')) {
      return 'bg-green-100 text-green-800 border-green-300';
    } else if (lowerCategory.includes('rotten') || lowerCategory.includes('bad')) {
      return 'bg-red-100 text-red-800 border-red-300';
    } else {
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    }
  };

  if (!user) {
    return null;
  }

  return (
    <DashboardLayout title="Freshness AI Scanner">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl p-6 mb-6">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
              <Sparkles className="h-8 w-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold mb-1">AI Freshness Scanner</h1>
              <p className="text-green-50">Upload images to detect fruit & vegetable freshness</p>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="card mb-6">
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Batch ID (Optional)
            </label>
            <input
              type="text"
              value={batchId}
              onChange={(e) => setBatchId(e.target.value)}
              placeholder="Enter batch ID to link scan results"
              className="input-field"
            />
          </div>

          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            
            {!previewUrl ? (
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">Click to upload an image</p>
                <p className="text-sm text-gray-500">PNG, JPG, or JPEG (Max 10MB)</p>
              </label>
            ) : (
              <div>
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded-lg shadow-md mb-4"
                />
                <div className="flex justify-center space-x-3">
                  <label htmlFor="file-upload" className="btn-secondary">
                    <Upload className="h-4 w-4 mr-2" />
                    Change Image
                  </label>
                  <button
                    onClick={handleScan}
                    disabled={isScanning}
                    className="btn-primary"
                  >
                    <Camera className="h-4 w-4 mr-2" />
                    {isScanning ? 'Analyzing...' : 'Scan Freshness'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            <div className="flex items-center">
              <XCircle className="h-5 w-5 mr-2" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900">Analysis Results</h2>
              <button onClick={handleReset} className="btn-secondary text-sm">
                Scan New Image
              </button>
            </div>

            {/* Check if product is NOT fresh */}
            {(result.freshnessCategory.toLowerCase().includes('rotten') || 
              result.freshnessCategory.toLowerCase().includes('bad') || 
              result.freshnessScore < 0.5) ? (
              // Product is NOT FRESH - Show UNFIT label only
              <div className="text-center p-12">
                <div className="flex justify-center mb-6">
                  <XCircle className="h-24 w-24 text-red-500" />
                </div>
                <div className="inline-block px-8 py-4 rounded-xl border-4 border-red-500 bg-red-100">
                  <h3 className="text-3xl font-bold text-red-800">BATCH UNFIT</h3>
                  <p className="text-red-700 mt-2">Product failed freshness inspection</p>
                </div>
                <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
                  <p className="text-red-900 font-semibold">
                    ⚠️ This batch does not meet freshness standards and should not be distributed.
                  </p>
                </div>
              </div>
            ) : (
              // Product IS FRESH - Show full details
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Category */}
                  <div className="text-center p-6 border-2 rounded-lg">
                    <div className="flex justify-center mb-3">
                      {getCategoryIcon(result.freshnessCategory)}
                    </div>
                    <div className={`inline-block px-4 py-2 rounded-lg border-2 font-semibold ${getCategoryColor(result.freshnessCategory)}`}>
                      {result.freshnessCategory}
                    </div>
                  </div>

                  {/* Scores */}
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-gray-700">Freshness Score</span>
                        <span className="font-bold text-gray-900">{(result.freshnessScore * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-green-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${result.freshnessScore * 100}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-gray-700">Confidence</span>
                        <span className="font-bold text-gray-900">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Message */}
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-blue-900">{result.message}</p>
                </div>

                {/* Technical Details */}
                {result.dominantClass && (
                  <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">Dominant Class:</span> {result.dominantClass}
                      {result.dominantScore && ` (${(result.dominantScore * 100).toFixed(1)}%)`}
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Info Section */}
        <div className="mt-6 bg-gradient-to-r from-blue-50 to-teal-50 rounded-lg p-6">
          <div className="flex items-start space-x-3">
            <Leaf className="h-6 w-6 text-teal-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">How it works</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Upload clear images of fruits or vegetables</li>
                <li>• AI model analyzes visual indicators of freshness</li>
                <li>• Get instant freshness scores and recommendations</li>
                <li>• Link results to batch IDs for quality tracking</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default FreshnessAIPage;
