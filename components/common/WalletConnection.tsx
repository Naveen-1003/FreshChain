'use client';

import React, { useState, useEffect } from 'react';
import { CheckCircle, Wifi, AlertCircle } from 'lucide-react';
import { getAuthService, UserRole } from '../../lib/services/authService';

interface BackendStatusProps {
  className?: string;
}

export const BackendStatus: React.FC<BackendStatusProps> = ({
  className = ''
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const authService = getAuthService();
  const user = authService.getUser();

  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}/health`);
      setIsConnected(response.ok);
    } catch (error) {
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className="flex items-center space-x-2 px-3 py-1 bg-yellow-100 rounded-lg">
          <div className="animate-spin h-4 w-4 border-2 border-yellow-600 border-t-transparent rounded-full"></div>
          <span className="text-xs text-yellow-700 font-medium">
            Checking...
          </span>
        </div>
      </div>
    );
  }

  if (!isConnected) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className="flex items-center space-x-2 px-3 py-1 bg-red-100 rounded-lg">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <span className="text-xs text-red-700 font-medium">
            Backend Offline
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 rounded-lg">
        <CheckCircle className="h-4 w-4 text-green-600" />
        <span className="text-xs text-green-700 font-medium">
          Backend Connected
        </span>
      </div>
      {user && (
        <div className="flex items-center space-x-2 px-3 py-1 bg-blue-100 rounded-lg">
          <span className="text-xs text-blue-700 font-medium">
            {user.username} ({user.role})
          </span>
          {user.role === UserRole.ADMIN && (
            <Wifi className="h-4 w-4 text-blue-600" title="MetaMask Required" />
          )}
        </div>
      )}
    </div>
  );
};

export default BackendStatus;