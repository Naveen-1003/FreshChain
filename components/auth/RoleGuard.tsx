'use client';

import React from 'react';
import { getAuthService, UserRole } from '../../lib/services/authService';

interface RoleGuardProps {
  allowedRoles: UserRole[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const RoleGuard: React.FC<RoleGuardProps> = ({ 
  allowedRoles, 
  children, 
  fallback = null 
}) => {
  const authService = getAuthService();
  const user = authService.getUser();

  if (!user || !allowedRoles.includes(user.role as UserRole)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

interface ActionButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  allowedRoles: UserRole[];
}

export const RoleBasedActionButton: React.FC<ActionButtonProps> = ({
  onClick,
  children,
  className = '',
  disabled = false,
  allowedRoles
}) => {
  const authService = getAuthService();
  const user = authService.getUser();

  // Don't render button if user doesn't have permission
  if (!user || !allowedRoles.includes(user.role as UserRole)) {
    return null;
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={className}
    >
      {children}
    </button>
  );
};

export default RoleGuard;