const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export enum UserRole {
  ADMIN = 'admin',
  PRODUCER = 'producer',
  AGGREGATOR = 'aggregator',
  RETAILER = 'retailer',
  TRANSPORTER = 'transporter',
  CONSUMER = 'consumer'
}

export interface User {
  id: string;
  username: string;
  role: UserRole;
  wallet_address?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
  walletAddress?: string; // For admin MetaMask login
}

export interface LoginResponse {
  success: boolean;
  token: string;
  user: User;
  message: string;
}

export class AuthService {
  private token: string | null = null;
  private user: User | null = null;

  constructor() {
    // Load from localStorage on initialization
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('freshchain_token');
      const userData = localStorage.getItem('freshchain_user');
      if (userData) {
        try {
          this.user = JSON.parse(userData);
        } catch (e) {
          console.error('Failed to parse user data:', e);
          this.clearAuth();
        }
      }
    }
  }

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      const response = await fetch(`${BACKEND_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      // Store auth data
      this.token = data.token;
      this.user = data.user;
      
      if (typeof window !== 'undefined') {
        localStorage.setItem('freshchain_token', data.token);
        localStorage.setItem('freshchain_user', JSON.stringify(data.user));
      }

      return data;
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Login failed');
    }
  }

  logout(): void {
    this.token = null;
    this.user = null;
    
    if (typeof window !== 'undefined') {
      localStorage.removeItem('freshchain_token');
      localStorage.removeItem('freshchain_user');
    }
  }

  getToken(): string | null {
    return this.token;
  }

  getUser(): User | null {
    return this.user;
  }

  isAuthenticated(): boolean {
    return this.token !== null && this.user !== null;
  }

  hasRole(role: UserRole): boolean {
    return this.user?.role === role;
  }

  canCreateBatch(): boolean {
    return (
      this.hasRole(UserRole.ADMIN) ||
      this.hasRole(UserRole.AGGREGATOR) ||
      this.hasRole(UserRole.PRODUCER)
    );
  }

  canUpdateStage(): boolean {
    return this.hasRole(UserRole.RETAILER) || this.hasRole(UserRole.TRANSPORTER);
  }

  canReportAlert(): boolean {
    return this.hasRole(UserRole.RETAILER) || this.hasRole(UserRole.TRANSPORTER);
  }

  needsMetaMask(): boolean {
    return this.hasRole(UserRole.ADMIN);
  }

  isConsumerOnly(): boolean {
    return this.hasRole(UserRole.CONSUMER);
  }

  getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  private clearAuth(): void {
    this.token = null;
    this.user = null;
    
    if (typeof window !== 'undefined') {
      localStorage.removeItem('freshchain_token');
      localStorage.removeItem('freshchain_user');
    }
  }
}

// Singleton instance
let authService: AuthService | null = null;

export const getAuthService = (): AuthService => {
  if (!authService) {
    authService = new AuthService();
  }
  return authService;
};

export default AuthService;