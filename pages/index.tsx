import { NextPage } from 'next';
import Head from 'next/head';
import Link from 'next/link';
import { Package, Eye, Shield, Zap, Camera, Blocks, CheckCircle, ArrowRight, Users, Lock } from 'lucide-react';

const HomePage: NextPage = () => {
  return (
    <>
      <Head>
        <title>FreshChain - Role-Based Supply Chain</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
              <Package className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">FreshChain</h1>
              <p className="text-sm text-gray-500 hidden sm:block">Role-Based Supply Chain</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link 
              href="/login" 
              className="flex items-center space-x-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors"
            >
              <Lock className="h-4 w-4" />
              <span>Login</span>
            </Link>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Hero Section */}
          <div className="bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl p-8 mb-8">
            <div className="text-center">
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
                Welcome to FreshChain
              </h1>
              <p className="text-xl text-white opacity-90 mb-8 max-w-3xl mx-auto">
                Role-based blockchain supply chain management with secure backend operations and wallet-free user experience
              </p>
              <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-6">
                <Link href="/login" className="bg-white text-emerald-600 hover:bg-gray-50 font-semibold py-4 px-8 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl flex items-center justify-center text-lg">
                  <Users className="h-6 w-6 mr-3" />
                  Get Started
                </Link>
                <Link href="/consumer-audit" className="bg-white bg-opacity-20 text-white hover:bg-white hover:text-emerald-600 font-semibold py-4 px-8 rounded-xl transition-all duration-200 flex items-center justify-center text-lg">
                  <Eye className="h-6 w-6 mr-3" />
                  Consumer Audit
                </Link>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 text-center">
              <div className="w-12 h-12 bg-emerald-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Users className="h-6 w-6 text-emerald-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Role-Based Access</h3>
              <p className="text-gray-500">Different interfaces for Admin, Retailer, Transporter, and Consumer roles</p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 text-center">
              <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Shield className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Backend Blockchain</h3>
              <p className="text-gray-500">All blockchain operations handled securely by backend system</p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 text-center">
              <div className="w-12 h-12 bg-yellow-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Zap className="h-6 w-6 text-yellow-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Wallet Required</h3>
              <p className="text-gray-500">Simple UI buttons - no MetaMask or crypto knowledge needed</p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 text-center">
              <div className="w-12 h-12 bg-purple-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Eye className="h-6 w-6 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">QR Code Scanning</h3>
              <p className="text-gray-500">Always shows latest blockchain state for consumer verification</p>
            </div>
          </div>

          {/* Call to Action */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 bg-emerald-50 border-emerald-200">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Ready to Get Started?</h2>
              <p className="text-gray-500 mb-6">
                Experience role-based supply chain management with FreshChain
              </p>
              <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-6">
                <Link href="/login" className="bg-emerald-500 hover:bg-emerald-600 text-white font-medium py-3 px-6 rounded-lg transition-all duration-200 shadow-sm hover:shadow-md flex items-center justify-center text-lg">
                  <Users className="h-6 w-6 mr-3" />
                  Login to Dashboard
                  <ArrowRight className="h-5 w-5 ml-3" />
                </Link>
                <Link href="/consumer-audit" className="bg-white hover:bg-gray-50 text-gray-700 font-medium py-3 px-6 rounded-lg border border-gray-200 transition-all duration-200 shadow-sm hover:shadow-md flex items-center justify-center text-lg">
                  <Camera className="h-6 w-6 mr-3" />
                  Consumer Audit
                  <ArrowRight className="h-5 w-5 ml-3" />
                </Link>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center mt-12 py-8 border-t border-gray-200">
            <p className="text-gray-500 mb-2">
              Role-based blockchain dApp for secure supply chain management
            </p>
            <div className="text-xs text-gray-400 space-y-1">
              <p>Backend API: {process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}</p>
              <p>Network: Shardeum Testnet via Backend</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default HomePage;