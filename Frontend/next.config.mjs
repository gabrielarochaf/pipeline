/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ["localhost", "127.0.0.1"], // Adicione 'localhost' e '127.0.0.1' aos domínios permitidos
  },
  // remotePatterns: [
  //   {
  //     protocol: "http",
  //     hostname: "localhost:3000/public",
  //     port: "",
  //     pathname: "/**",
  //   },
  // ],
};

export default nextConfig;
