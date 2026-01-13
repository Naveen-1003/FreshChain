/**
 * Backend Service for FreshChain
 * Communicates with FastAPI backend that handles all blockchain operations
 * Frontend never directly interacts with blockchain
 */

import { getAuthService } from './authService';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export enum BatchStage {
  CREATED = 'Created',
  HARVESTED = 'Harvested',
  IN_TRANSIT = 'In Transit',
  AT_RETAILER = 'At Retailer',
  SELLING = 'Selling'
}

export enum SensorType {
  TRANSPORTER = 'transporter',
  RETAILER = 'retailer'
}

export interface BatchCreationRequest {
  batchId: string;
  productType: string;
  sampleImageBase64?: string;
  freshnessScore?: number;
  freshnessCategory?: string;
}

export interface BatchUpdateRequest {
  batchId: string;
  stage: BatchStage;
  location: string;
}

export interface ReportAlertRequest {
  batchId: string;
  alertType: string;
  encryptedData: string;
}

export interface LocationUpdate {
  stage: BatchStage;
  location: string;
  timestamp: string;
  transactionHash: string;
  updatedBy: string;
}

export interface Alert {
  alertType: string;
  encryptedData: string;
  timestamp: string;
  transactionHash: string;
}

export interface BatchDetails {
  batchId: string;
  productType: string;
  created: string;
  currentStage: BatchStage;
  currentLocation: string;
  locationHistory: LocationUpdate[];
  alerts: Alert[];
  isActive: boolean;
  isFinalStage: boolean;
  // Time and freshness data for consumers
  ageInDays?: number;
  ageInHours?: number;
  estimatedShelfLifeDays?: number;
  freshnessScore?: number;
  freshnessCategory?: string;
}

export interface QRLinkRequest {
  batchId: string;
  sensorId: string;
  locationType: SensorType;
  qrPayload?: string;
  sampleImageBase64?: string;  // Mandatory sample testing image
  freshnessScore?: number;
  freshnessCategory?: string;
}

export interface SensorRegistration {
  sensorId: string;
  batchId: string;
  sensorType: SensorType;
}

export interface SensorDataSubmission {
  batchId: string;
  sensorId: string;
  temperature: number;
  humidity: number;
}

export interface SensorReading {
  batchId: string;
  sensorId: string;
  sensorType: string;
  temperature: number;
  humidity: number;
  timestamp: string;
  source: string;
}

export interface SensorReadingsResponse {
  readings: SensorReading[];
}

export interface SensorInfo {
  sensorId: string;
  sensorType: SensorType;
  label?: string;
  vehicleOrStoreId?: string;
  owner: string;
  registeredAt: string;
  isLinked: boolean;
  currentBatch?: string;
}

export interface FreshnessResult {
  batchId?: string;
  freshnessScore: number;
  freshnessCategory: string;
  confidence: number;
  message: string;
  dominantClass?: string;
  dominantScore?: number;
}

export interface ParentOfferPayload {
  productType: string;
  unit: string;
  basePrice: number;
  totalQuantity: number;
  currency?: string;
  metadata?: Record<string, any>;
}

export interface ParentOffer extends ParentOfferPayload {
  parentId: string;
  parentBatchNumber: string;
  producer: string;
  availableQuantity: number;
  pricingCurrency: string;
  createdAt: string;
  status: 'draft' | 'published';
  publishedAt?: string;
  metadata?: Record<string, any>;
}

export interface MarketplacePaymentInfo {
  orderId: string;
  amount: number;
  currency: string;
  status: string;
  createdAt: string;
  paymentId?: string;
  paidAt?: string;
}

export interface MarketplaceRequest {
  requestId: string;
  parentId: string;
  parentBatchNumber?: string;
  parentProductType?: string;
  retailer: string;
  producer?: string;
  quantity: number;
  bidPrice: number;
  status: string;
  createdAt: string;
  approvedAt?: string;
  currency: string;
  advancePercent: number;
  payment?: MarketplacePaymentInfo | null;
  childBatchId?: string;
  fulfilledAt?: string;
}

export interface RetailerBidPayload {
  parentId: string;
  quantity: number;
  bidPrice: number;
}

export interface PaymentOrderResult {
  orderId: string;
  amount: number;
  currency: string;
  order: Record<string, any>;
}

export interface PaymentConfirmationPayload {
  paymentId: string;
  orderId: string;
}

export interface FulfillBidPayload {
  childBatchId: string;
  productType?: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  transactionHash?: string;
}

export class BackendService {
  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    try {
      const authService = getAuthService();
      const headers = authService.getAuthHeaders();

      // Add timeout for faster failure
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      try {
        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
          headers: {
            ...headers,
            ...options.headers,
          },
          signal: controller.signal,
          ...options,
        });

        clearTimeout(timeoutId);

        const data = await response.json();

        if (!response.ok) {
          return {
            success: false,
            error: data.detail || `HTTP ${response.status}: ${response.statusText}`,
          };
        }

        // Handle different response structures
        if (data.success !== undefined) {
          // Backend returns {success: true, data: [...]}
          return {
            success: data.success,
            data: data.data,
            message: data.message,
            transactionHash: data.transactionHash,
          };
        } else {
          // Backend returns data directly
          return {
            success: true,
            data: data,
            message: data.message,
          transactionHash: data.transactionHash,
        };
      }
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === 'AbortError') {
          return {
            success: false,
            error: 'Request timeout - please try again',
          };
        }
        throw fetchError;
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error occurred',
      };
    }
  }

  /**
   * Create a new batch (Admin or Retailer only)
   * Backend signs the transaction with system wallet
   */
  async createBatch(request: BatchCreationRequest): Promise<ApiResponse<void>> {
    return this.makeRequest('/batch/create', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Update batch stage (Retailer or Transporter only)
   * Backend signs the transaction with system wallet
   */
  async updateBatchStage(request: BatchUpdateRequest): Promise<ApiResponse<void>> {
    return this.makeRequest('/batch/update-stage', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Report an alert/excursion for a batch (Retailer or Transporter only)
   */
  async reportAlert(request: ReportAlertRequest): Promise<ApiResponse<void>> {
    return this.makeRequest('/batch/report-alert', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get batch details (Public - used for QR scanning)
   * Always returns the latest blockchain state
   */
  async getBatchDetails(batchId: string): Promise<ApiResponse<BatchDetails>> {
    return this.makeRequest<BatchDetails>(`/batch/${batchId}`);
  }

  /**
   * Check backend health and blockchain connection
   */
  async checkHealth(): Promise<ApiResponse<{ status: string; blockchain_connected: boolean; network: string }>> {
    return this.makeRequest<{ status: string; blockchain_connected: boolean; network: string }>('/health');
  }

  /**
   * Get all batches (Admin only)
   */
  async getAllBatches(): Promise<ApiResponse<BatchDetails[]>> {
    return this.makeRequest<BatchDetails[]>('/admin/batches');
  }

  /**
   * Get batches filtered by stage (Supply chain roles)
   * Transporters see In Transit batches, Retailers see At Retailer batches
   */
  async getBatchesByStage(stage?: string): Promise<ApiResponse<BatchDetails[]>> {
    try {
      const url = stage ? `/batches/by-stage?stage=${encodeURIComponent(stage)}` : '/batches/by-stage';
      const token = getAuthService().getToken();
      const response = await fetch(`${BACKEND_URL}${url}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch batches: ${response.statusText}`);
      }

      const result = await response.json();
      // Backend returns { success: true, data: [...] }
      return { success: true, data: result.data || [] };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch batches',
      };
    }
  }

  /**
   * Link a retailer/transporter sensor to a batch by scanning QR payload
   */
  async linkSensorToBatch(request: QRLinkRequest): Promise<ApiResponse<any>> {
    return this.makeRequest('/qr/scan', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Register a new sensor
   */
  async registerSensor(request: SensorRegistration): Promise<ApiResponse<any>> {
    return this.makeRequest('/sensors/register', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Submit sensor data readings
   */
  async submitSensorData(request: SensorDataSubmission): Promise<ApiResponse<any>> {
    return this.makeRequest('/sensors/data', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get sensor readings for a batch
   */
  async getSensorReadings(batchId: string): Promise<ApiResponse<SensorReadingsResponse>> {
    return this.makeRequest<SensorReadingsResponse>(`/sensors/batch/${batchId}`);
  }

  /**
   * Get list of available sensors
   */
  async getAvailableSensors(sensorType?: SensorType): Promise<ApiResponse<{ sensors: SensorInfo[] }>> {
    const query = sensorType ? `?sensorType=${sensorType}` : '';
    return this.makeRequest<{ sensors: SensorInfo[] }>(`/sensors/available${query}`);
  }

  /**
   * Scan image for freshness detection
   */
  async scanFreshness(imageFile: File, batchId?: string): Promise<ApiResponse<FreshnessResult>> {
    const formData = new FormData();
    formData.append('file', imageFile);
    if (batchId) {
      formData.append('batchId', batchId);
    }

    const authService = getAuthService();
    const token = authService.getToken();
    
    const response = await fetch(`${BACKEND_URL}/ml/freshness-scan`, {
      method: 'POST',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to scan image' }));
      return {
        success: false,
        error: errorData.detail || 'Failed to scan image',
      };
    }

    const data = await response.json();
    return {
      success: true,
      data,
    };
  }

  /** Marketplace helpers **/
  async listParentOffers(status?: string): Promise<ApiResponse<ParentOffer[]>> {
    const query = status ? `?status=${encodeURIComponent(status)}` : '';
    return this.makeRequest<ParentOffer[]>(`/marketplace/parent${query}`);
  }

  async createParentOffer(payload: ParentOfferPayload): Promise<ApiResponse<ParentOffer>> {
    return this.makeRequest<ParentOffer>('/marketplace/parent', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async publishParentOffer(parentId: string): Promise<ApiResponse<ParentOffer>> {
    return this.makeRequest<ParentOffer>(`/marketplace/parent/${parentId}/publish`, {
      method: 'POST',
    });
  }

  async listMarketplaceRequests(parentId?: string): Promise<ApiResponse<MarketplaceRequest[]>> {
    const query = parentId ? `?parentId=${encodeURIComponent(parentId)}` : '';
    return this.makeRequest<MarketplaceRequest[]>(`/marketplace/requests${query}`);
  }

  async createMarketplaceRequest(payload: RetailerBidPayload): Promise<ApiResponse<MarketplaceRequest>> {
    return this.makeRequest<MarketplaceRequest>('/marketplace/requests', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async approveMarketplaceRequest(requestId: string): Promise<ApiResponse<MarketplaceRequest>> {
    return this.makeRequest<MarketplaceRequest>(`/marketplace/requests/${requestId}/approve`, {
      method: 'POST',
    });
  }

  async rejectMarketplaceRequest(requestId: string): Promise<ApiResponse<MarketplaceRequest>> {
    return this.makeRequest<MarketplaceRequest>(`/marketplace/requests/${requestId}/reject`, {
      method: 'POST',
    });
  }

  async createMarketplacePaymentOrder(requestId: string): Promise<ApiResponse<PaymentOrderResult>> {
    return this.makeRequest<PaymentOrderResult>(`/marketplace/requests/${requestId}/order`, {
      method: 'POST',
    });
  }

  async confirmMarketplacePayment(
    requestId: string,
    payload: PaymentConfirmationPayload,
  ): Promise<ApiResponse<MarketplaceRequest>> {
    return this.makeRequest<MarketplaceRequest>(`/marketplace/requests/${requestId}/confirm`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async fulfillMarketplaceRequest(
    requestId: string,
    payload: FulfillBidPayload,
  ): Promise<ApiResponse<void>> {
    return this.makeRequest(`/marketplace/requests/${requestId}/fulfill`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Admin MetaMask operations (simplified - no signature modals)
  async adminCreateBatchWithMetaMask(request: BatchCreationRequest): Promise<ApiResponse<void>> {
    return this.makeRequest('/admin/metamask/create-batch', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async adminUpdateStageWithMetaMask(request: BatchUpdateRequest): Promise<ApiResponse<void>> {
    return this.makeRequest('/admin/metamask/update-stage', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getWalletStatus(): Promise<ApiResponse<any>> {
    return this.makeRequest('/admin/wallet/status');
  }
}

// Singleton instance
let backendService: BackendService | null = null;

export const getBackendService = (): BackendService => {
  if (!backendService) {
    backendService = new BackendService();
  }
  return backendService;
};

export default BackendService;