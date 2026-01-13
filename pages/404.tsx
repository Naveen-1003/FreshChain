import { NextPage } from 'next';
import Link from 'next/link';
import { Search, Home, Package, Eye } from 'lucide-react';

const NotFoundPage: NextPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-green-50 flex items-center justify-center p-4">
      <div className="max-w-lg w-full text-center">
        <div className="w-20 h-20 bg-emerald-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <Search className="h-10 w-10 text-emerald-600" />
        </div>
        
        <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Page Not Found</h2>
        
        <p className="text-gray-600 mb-8">
          The page you're looking for doesn't exist. It might have been moved, deleted, or you entered the wrong URL.
        </p>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <Link
            href="/retailer"
            className="flex items-center justify-center bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-emerald-600 hover:to-teal-600 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <Package className="h-5 w-5 mr-2" />
            Retailer Dashboard
          </Link>
          
          <Link
            href="/consumer-audit"
            className="flex items-center justify-center bg-white text-emerald-600 border-2 border-emerald-500 py-3 px-6 rounded-xl font-semibold hover:bg-emerald-50 transition-all duration-200"
          >
            <Eye className="h-5 w-5 mr-2" />
            Consumer Audit
          </Link>
        </div>
        
        <Link
          href="/"
          className="inline-flex items-center justify-center bg-white text-gray-700 py-3 px-6 rounded-xl font-medium border border-gray-200 hover:bg-gray-50 transition-colors"
        >
          <Home className="h-5 w-5 mr-2" />
          Back to Home
        </Link>
        
        <div className="mt-8 text-sm text-gray-500">
          <p>FreshChain - Blockchain Supply Chain Tracking</p>
        </div>
      </div>
    </div>
  );
};

export default NotFoundPage;