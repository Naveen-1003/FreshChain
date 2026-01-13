import { NextPage } from 'next';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getAuthService } from '../lib/services/authService';
import LoginForm from '../components/auth/LoginForm';

const LoginPage: NextPage = () => {
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const authService = getAuthService();

  useEffect(() => {
    // Check if user is already logged in
    if (authService.isAuthenticated()) {
      router.push('/dashboard');
    } else {
      setIsLoading(false);
    }
  }, [router]);

  const handleLogin = (user: any) => {
    // Redirect based on user role or to dashboard
    router.push('/dashboard');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-emerald-500 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return <LoginForm onLogin={handleLogin} />;
};

export default LoginPage;