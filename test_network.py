#!/usr/bin/env python3
"""
Test network connectivity to Supabase
"""

import socket
import requests
import asyncio

def test_dns_resolution():
    """Test if we can resolve the hostname."""
    try:
        host = "db.bgmylhcqhsrvmmlqjhxo.supabase.co"
        ip = socket.gethostbyname(host)
        print(f"✅ DNS resolution successful: {host} -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return False

def test_http_connection():
    """Test HTTP connection to Supabase."""
    try:
        url = "https://bgmylhcqhsrvmmlqjhxo.supabase.co"
        response = requests.get(url, timeout=10)
        print(f"✅ HTTP connection successful: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ HTTP connection failed: {e}")
        return False

def test_port_connection():
    """Test if port 5432 is reachable."""
    try:
        host = "db.bgmylhcqhsrvmmlqjhxo.supabase.co"
        port = 5432
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is reachable")
            return True
        else:
            print(f"❌ Port {port} is not reachable (error code: {result})")
            return False
    except Exception as e:
        print(f"❌ Port test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing network connectivity to Supabase...")
    print()
    
    dns_ok = test_dns_resolution()
    http_ok = test_http_connection()
    port_ok = test_port_connection()
    
    print()
    if dns_ok and http_ok and port_ok:
        print("✅ All network tests passed - connection should work!")
        print("🔧 The issue might be with credentials or database configuration")
    else:
        print("❌ Network connectivity issues detected")
        print("🔧 Try: different network, VPN, or check firewall settings")
