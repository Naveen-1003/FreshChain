import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { params } = req.query;
  const [width = '300', height = '200'] = Array.isArray(params) ? params : [params];
  
  // Create a simple SVG placeholder
  const svg = `
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="#e5e7eb"/>
      <rect x="20%" y="20%" width="60%" height="60%" fill="#10b981" rx="8"/>
      <text x="50%" y="45%" text-anchor="middle" fill="white" font-family="Arial" font-size="14">
        Apple Crate
      </text>
      <rect x="70%" y="70%" width="25%" height="25%" fill="white" rx="4"/>
      <text x="82.5%" y="85%" text-anchor="middle" fill="#10b981" font-family="Arial" font-size="8">
        QR
      </text>
    </svg>
  `;

  res.setHeader('Content-Type', 'image/svg+xml');
  res.setHeader('Cache-Control', 'public, max-age=31536000');
  res.status(200).send(svg);
}