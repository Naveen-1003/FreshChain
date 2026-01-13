import { NextPage } from 'next';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

const RetailerPage: NextPage = () => {
  const router = useRouter();

  useEffect(() => {
    // Redirect to professional page as retailer functionality is now role-based
    router.push('/professional');
  }, [router]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Redirecting...</h2>
        <p className="text-gray-600">Taking you to the professional dashboard.</p>
      </div>
    </div>
  );
};

export default RetailerPage;