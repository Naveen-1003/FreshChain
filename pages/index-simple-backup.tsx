import { NextPage } from 'next';
import Head from 'next/head';
import Link from 'next/link';

const SimpleHomePage: NextPage = () => {
  return (
    <>
      <Head>
        <title>FreshChain - Simple</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              FreshChain
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Role-based blockchain supply chain management
            </p>
            
            <div className="space-y-4 max-w-md mx-auto">
              <Link 
                href="/login"
                className="block bg-emerald-500 hover:bg-emerald-600 text-white font-medium py-3 px-6 rounded-lg transition-colors"
              >
                Login to Dashboard
              </Link>
              
              <Link 
                href="/consumer-audit"
                className="block bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-6 rounded-lg transition-colors"
              >
                Consumer Audit
              </Link>
              
              <Link 
                href="/test"
                className="block bg-gray-500 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-lg transition-colors"
              >
                Test Page
              </Link>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default SimpleHomePage;