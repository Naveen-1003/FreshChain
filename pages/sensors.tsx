import { NextPage } from 'next';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Thermometer, Droplets, Wifi, Plus, BarChart3, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import DashboardLayout from '../components/layout/DashboardLayout';
import { getAuthService, UserRole } from '../lib/services/authService';
import { getBackendService, SensorType } from '../lib/services/backendService';

interface SensorReading {
  batchId: string;
  sensorId: string;
  sensorType: string;
  temperature: number;
  humidity: number;
  timestamp: string;
  source: string;
  temperatureStatus?: string; // "Normal", "Too Low", or "Too High"
}

const SensorsPage: NextPage = () => {
  const [activeTab, setActiveTab] = useState<'register' | 'submit' | 'view'>('register');
  
  // Registration form
  const [regSensorId, setRegSensorId] = useState('');
  const [regBatchId, setRegBatchId] = useState('');
  const [regType, setRegType] = useState<SensorType>(SensorType.TRANSPORTER);
  
  // Data submission form
  const [submitBatchId, setSubmitBatchId] = useState('');
  const [submitSensorId, setSubmitSensorId] = useState('');
  const [submitTemp, setSubmitTemp] = useState('');
  const [submitHumidity, setSubmitHumidity] = useState('');
  
  // View data
  const [viewBatchId, setViewBatchId] = useState('');
  const [readings, setReadings] = useState<SensorReading[]>([]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const router = useRouter();
  const authService = getAuthService();
  const backendService = getBackendService();
  const user = authService.getUser();

  useEffect(() => {
    if (!authService.isAuthenticated()) {
      router.push('/login');
    }
  }, [router]);

  const handleRegisterSensor = async () => {
    if (!regSensorId.trim() || !regBatchId.trim()) {
      setError('Please provide sensor ID and batch ID');
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.registerSensor({
        sensorId: regSensorId.trim(),
        batchId: regBatchId.trim(),
        sensorType: regType,
      });

      if (response.success) {
        setSuccess(`Sensor ${regSensorId} registered successfully!`);
        setRegSensorId('');
        setRegBatchId('');
      } else {
        setError(response.error || 'Failed to register sensor');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to register sensor');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitData = async () => {
    if (!submitBatchId.trim() || !submitSensorId.trim() || !submitTemp || !submitHumidity) {
      setError('Please fill in all fields');
      return;
    }

    const temp = parseFloat(submitTemp);
    const humidity = parseFloat(submitHumidity);

    if (isNaN(temp) || isNaN(humidity)) {
      setError('Temperature and humidity must be valid numbers');
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.submitSensorData({
        batchId: submitBatchId.trim(),
        sensorId: submitSensorId.trim(),
        temperature: temp,
        humidity: humidity,
      });

      if (response.success) {
        setSuccess('Sensor data submitted successfully!');
        setSubmitTemp('');
        setSubmitHumidity('');
      } else {
        setError(response.error || 'Failed to submit sensor data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit sensor data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleViewReadings = async () => {
    if (!viewBatchId.trim()) {
      setError('Please enter a batch ID');
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess('');
    setReadings([]);

    try {
      const response = await backendService.getSensorReadings(viewBatchId.trim());

      if (response.success && response.data) {
        setReadings(response.data.readings || []);
        if (response.data.readings?.length === 0) {
          setError('No sensor readings found for this batch');
        }
      } else {
        setError(response.error || 'Failed to fetch sensor readings');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sensor readings');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getTemperatureColor = (temp: number) => {
    if (temp < 5) return 'text-blue-600';
    if (temp < 15) return 'text-green-600';
    if (temp < 25) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHumidityColor = (humidity: number) => {
    if (humidity < 40) return 'text-red-600';
    if (humidity < 60) return 'text-green-600';
    if (humidity < 80) return 'text-yellow-600';
    return 'text-blue-600';
  };

  if (!user) {
    return null;
  }

  const canManageSensors = user.role === UserRole.ADMIN || 
                           user.role === UserRole.TRANSPORTER || 
                           user.role === UserRole.RETAILER ||
                           user.role === UserRole.PRODUCER ||
                           user.role === UserRole.AGGREGATOR;

  if (!canManageSensors) {
    return (
      <DashboardLayout title="Sensor Dashboard">
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Access Restricted</h2>
            <p className="text-gray-600">You don't have permission to manage sensors</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Sensor Dashboard">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl p-6 mb-6">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
              <Activity className="h-8 w-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold mb-1">IoT Sensor Dashboard</h1>
              <p className="text-blue-50">Monitor temperature and humidity across your supply chain</p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="card mb-6">
          <div className="flex space-x-2 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('register')}
              className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
                activeTab === 'register'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <Wifi className="h-4 w-4 inline mr-2" />
              Register Sensor
            </button>
            <button
              onClick={() => setActiveTab('submit')}
              className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
                activeTab === 'submit'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <Plus className="h-4 w-4 inline mr-2" />
              Submit Data
            </button>
            <button
              onClick={() => setActiveTab('view')}
              className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
                activeTab === 'view'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <BarChart3 className="h-4 w-4 inline mr-2" />
              View Readings
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {/* Register Sensor Tab */}
            {activeTab === 'register' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sensor ID</label>
                  <input
                    type="text"
                    value={regSensorId}
                    onChange={(e) => setRegSensorId(e.target.value)}
                    placeholder="e.g., SENSOR-001"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Batch ID</label>
                  <input
                    type="text"
                    value={regBatchId}
                    onChange={(e) => setRegBatchId(e.target.value)}
                    placeholder="Enter batch ID to link sensor"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sensor Type</label>
                  <select
                    value={regType}
                    onChange={(e) => setRegType(e.target.value as SensorType)}
                    className="input-field"
                  >
                    <option value={SensorType.TRANSPORTER}>Transporter</option>
                    <option value={SensorType.RETAILER}>Retailer</option>
                  </select>
                </div>

                <button
                  onClick={handleRegisterSensor}
                  disabled={isLoading}
                  className="btn-primary"
                >
                  <Wifi className="h-4 w-4 mr-2" />
                  {isLoading ? 'Registering...' : 'Register Sensor'}
                </button>
              </div>
            )}

            {/* Submit Data Tab */}
            {activeTab === 'submit' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Batch ID</label>
                  <input
                    type="text"
                    value={submitBatchId}
                    onChange={(e) => setSubmitBatchId(e.target.value)}
                    placeholder="Enter batch ID"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sensor ID</label>
                  <input
                    type="text"
                    value={submitSensorId}
                    onChange={(e) => setSubmitSensorId(e.target.value)}
                    placeholder="e.g., SENSOR-001"
                    className="input-field"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <Thermometer className="h-4 w-4 inline mr-1" />
                      Temperature (°C)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={submitTemp}
                      onChange={(e) => setSubmitTemp(e.target.value)}
                      placeholder="e.g., 4.5"
                      className="input-field"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <Droplets className="h-4 w-4 inline mr-1" />
                      Humidity (%)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={submitHumidity}
                      onChange={(e) => setSubmitHumidity(e.target.value)}
                      placeholder="e.g., 65.0"
                      className="input-field"
                    />
                  </div>
                </div>

                <button
                  onClick={handleSubmitData}
                  disabled={isLoading}
                  className="btn-primary"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {isLoading ? 'Submitting...' : 'Submit Reading'}
                </button>
              </div>
            )}

            {/* View Readings Tab */}
            {activeTab === 'view' && (
              <div className="space-y-4">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={viewBatchId}
                    onChange={(e) => setViewBatchId(e.target.value)}
                    placeholder="Enter batch ID"
                    className="input-field flex-1"
                  />
                  <button
                    onClick={handleViewReadings}
                    disabled={isLoading}
                    className="btn-primary"
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    {isLoading ? 'Loading...' : 'View Readings'}
                  </button>
                </div>

                {readings.length > 0 && (
                  <div className="mt-6 space-y-3">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {readings.length} Reading{readings.length !== 1 ? 's' : ''} Found
                    </h3>
                    {readings.map((reading, idx) => (
                      <div key={idx} className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-xs text-gray-500 mb-1">Sensor ID</p>
                            <p className="font-medium text-gray-900">{reading.sensorId}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">Temperature</p>
                            <div className="flex items-center gap-2">
                              <p className={`font-bold text-lg ${getTemperatureColor(reading.temperature)}`}>
                                <Thermometer className="h-4 w-4 inline mr-1" />
                                {reading.temperature.toFixed(1)}°C
                              </p>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${
                                reading.temperatureStatus === 'Normal' 
                                  ? 'bg-green-100 text-green-700'
                                  : reading.temperatureStatus === 'Too Low'
                                  ? 'bg-blue-100 text-blue-700'
                                  : 'bg-red-100 text-red-700'
                              }`}>
                                {reading.temperatureStatus || 'Normal'}
                              </span>
                            </div>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">Humidity</p>
                            <p className={`font-bold text-lg ${getHumidityColor(reading.humidity)}`}>
                              <Droplets className="h-4 w-4 inline mr-1" />
                              {reading.humidity.toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">Timestamp</p>
                            <p className="text-sm text-gray-700">{formatDate(reading.timestamp)}</p>
                          </div>
                        </div>
                        <div className="mt-2 flex items-center text-xs text-gray-500">
                          <span className="px-2 py-1 bg-gray-100 rounded">
                            {reading.sensorType}
                          </span>
                          <span className="ml-2">Source: {reading.source}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Alerts */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 mr-2" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {success && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg mb-4">
            <div className="flex items-center">
              <CheckCircle className="h-5 w-5 mr-2" />
              <span>{success}</span>
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
          <div className="flex items-start space-x-3">
            <Activity className="h-6 w-6 text-indigo-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Sensor Management Guide</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Register sensors to link them with specific batches</li>
                <li>• Submit real-time temperature and humidity readings</li>
                <li>• View historical sensor data for quality monitoring</li>
                <li>• Use transporter sensors during transit, retailer sensors at storage</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default SensorsPage;
