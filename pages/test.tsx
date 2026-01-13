import { NextPage } from 'next';
import Head from 'next/head';
import Link from 'next/link';

const TestPage: NextPage = () => {
  const handleClick = () => {
    alert('Button clicked!');
  };

  return (
    <>
      <Head>
        <title>Test Page</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
          <h1 className="text-2xl font-bold mb-4">Test Page</h1>
          <p className="mb-4">This is a simple test page to check if clicking works.</p>
          
          <div className="space-y-4">
            <button 
              onClick={handleClick}
              className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition-colors"
            >
              Test Button (Alert)
            </button>
            
            <Link 
              href="/"
              className="block w-full bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg transition-colors text-center"
            >
              Back to Home
            </Link>
            
            <Link 
              href="/login"
              className="block w-full bg-purple-500 hover:bg-purple-600 text-white font-medium py-2 px-4 rounded-lg transition-colors text-center"
            >
              Go to Login
            </Link>
          </div>
        </div>
      </div>
    </>
  );
};

export default TestPage;