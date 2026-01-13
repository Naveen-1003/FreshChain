import { NextPage } from 'next';
import Link from 'next/link';
import { AlertTriangle, Home, RefreshCw } from 'lucide-react';

const ServerErrorPage: NextPage = () => {
  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-green-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full text-center">
        <div className="w-20 h-20 bg-red-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <AlertTriangle className="h-10 w-10 text-red-600" />
        </div>
        
        <h1 className="text-6xl font-bold text-gray-900 mb-4">500</h1>
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Server Error</h2>
        
        <p className="text-gray-600 mb-8">
          Something went wrong on our end. We're working to fix this issue. Please try again in a few moments.
        </p>
        
        <div className="space-y-4">
          <button
            onClick={handleRefresh}
            className="inline-flex items-center justify-center w-full bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-emerald-600 hover:to-teal-600 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <RefreshCw className="h-5 w-5 mr-2" />
            Try Again
          </button>
          
          <Link
            href="/"
            className="inline-flex items-center justify-center w-full bg-white text-gray-700 py-3 px-6 rounded-xl font-medium border border-gray-200 hover:bg-gray-50 transition-colors"
          >
            <Home className="h-5 w-5 mr-2" />
            Go Home
          </Link>
        </div>
        
        <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-700">
            If this problem persists, please contact support or try accessing the system later.
          </p>
        </div>
        
        <div className="mt-6 text-sm text-gray-500">
          <p>FreshChain - Blockchain Supply Chain Tracking</p>
        </div>
      </div>
    </div>
  );
};

export default ServerErrorPage;