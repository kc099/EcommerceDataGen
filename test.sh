#!/bin/bash

# Port 1883 Investigation and Fix Script

echo "🔍 INVESTIGATING PORT 1883 CONFLICT"
echo "==================================="

echo
echo "1. 🔍 What's Actually Using Port 1883?"
echo "------------------------------------"

echo "📊 Checking with netstat..."
NETSTAT_RESULT=$(sudo netstat -tulpn | grep :1883)
if [ -n "$NETSTAT_RESULT" ]; then
    echo "Found process using port 1883:"
    echo "$NETSTAT_RESULT"
    
    # Extract PID from netstat output
    PID=$(echo "$NETSTAT_RESULT" | awk '{print $NF}' | cut -d'/' -f1)
    if [ "$PID" != "-" ] && [ -n "$PID" ]; then
        echo "📋 Process details for PID $PID:"
        ps -p "$PID" -o pid,ppid,cmd,user,start 2>/dev/null || echo "Process not found in ps"
    fi
else
    echo "❌ No process found using port 1883 with netstat"
fi

echo
echo "📊 Checking with ss..."
SS_RESULT=$(sudo ss -tulpn | grep :1883)
if [ -n "$SS_RESULT" ]; then
    echo "Found process using port 1883:"
    echo "$SS_RESULT"
else
    echo "❌ No process found using port 1883 with ss"
fi

echo
echo "📊 Checking with lsof..."
LSOF_RESULT=$(sudo lsof -i :1883 2>/dev/null)
if [ -n "$LSOF_RESULT" ]; then
    echo "Found process using port 1883:"
    echo "$LSOF_RESULT"
else
    echo "❌ No process found using port 1883 with lsof"
fi

echo
echo "📊 Checking IPv6 as well..."
IPV6_RESULT=$(sudo netstat -tulpn | grep :::1883)
if [ -n "$IPV6_RESULT" ]; then
    echo "Found IPv6 process using port 1883:"
    echo "$IPV6_RESULT"
else
    echo "❌ No IPv6 process found using port 1883"
fi

echo
echo "2. 🔍 Check All Mosquitto-Related Processes"
echo "------------------------------------------"

echo "📊 All processes with 'mosquitto' in name or command:"
ps aux | grep -i mosquitto | grep -v grep || echo "No mosquitto processes found"

echo
echo "📊 All processes listening on any port 188*:"
sudo netstat -tulpn | grep :188 || echo "No processes found on 188* ports"

echo
echo "3. 🔍 Check Systemd Socket Units"
echo "-------------------------------"

echo "📊 Checking for systemd socket units that might be holding the port:"
sudo systemctl list-units --type=socket | grep -i mqtt || echo "No MQTT socket units found"
sudo systemctl list-units --type=socket | grep 1883 || echo "No socket units on port 1883 found"

echo
echo "4. 🔍 Check System Configuration"
echo "------------------------------"

echo "📊 Mosquitto configuration file:"
if [ -f "/etc/mosquitto/mosquitto.conf" ]; then
    echo "Found mosquitto.conf"
    echo "Port configuration:"
    grep -n "^listener\|^port" /etc/mosquitto/mosquitto.conf 2>/dev/null || echo "No explicit port configuration found"
else
    echo "❌ No mosquitto.conf found"
fi

echo
echo "📊 Check if there are other MQTT brokers installed:"
which emqx 2>/dev/null && echo "✅ EMQX found" || echo "❌ EMQX not found"
which vernemq 2>/dev/null && echo "✅ VerneMQ found" || echo "❌ VerneMQ not found"
which activemq 2>/dev/null && echo "✅ ActiveMQ found" || echo "❌ ActiveMQ not found"

echo
echo "📊 Check for Docker containers using port 1883:"
if command -v docker >/dev/null 2>&1; then
    DOCKER_CONTAINERS=$(sudo docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 1883 || echo "No Docker containers using port 1883")
    echo "$DOCKER_CONTAINERS"
else
    echo "Docker not installed"
fi

echo
echo "5. 🛠️ ATTEMPTED FIXES"
echo "==================="

echo
echo "Fix 1: Force kill anything on port 1883..."
sudo fuser -k 1883/tcp 2>/dev/null && echo "✅ Killed processes on port 1883" || echo "ℹ️  No processes to kill on port 1883"

echo
echo "Fix 2: Stop any MQTT-related services..."
sudo systemctl stop mosquitto 2>/dev/null && echo "✅ Stopped mosquitto" || echo "ℹ️  Mosquitto already stopped"

# Check for other common MQTT services
for service in emqx vernemq activemq; do
    if systemctl is-active "$service" >/dev/null 2>&1; then
        sudo systemctl stop "$service" && echo "✅ Stopped $service" || echo "❌ Failed to stop $service"
    fi
done

echo
echo "Fix 3: Wait for port to be completely released..."
sleep 5

echo
echo "Fix 4: Check if port is now free..."
if sudo netstat -tulpn | grep :1883 >/dev/null || sudo ss -tulpn | grep :1883 >/dev/null; then
    echo "❌ Port 1883 is STILL in use!"
    echo "🔍 Current process using port:"
    sudo lsof -i :1883 2>/dev/null || sudo netstat -tulpn | grep :1883
    
    echo
    echo "🚨 AGGRESSIVE FIX NEEDED:"
    echo "========================"
    echo "The following commands will forcefully free port 1883:"
    echo
    
    # Get the PID using the port
    PORT_PID=$(sudo lsof -t -i :1883 2>/dev/null | head -1)
    if [ -n "$PORT_PID" ]; then
        echo "# Kill the specific process:"
        echo "sudo kill -9 $PORT_PID"
        echo
        echo "Would you like me to kill PID $PORT_PID? (y/n)"
        # Don't actually kill automatically - let user decide
    fi
    
    echo "# Alternative - kill everything on port 1883:"
    echo "sudo fuser -k -9 1883/tcp"
    echo
    echo "# Or reboot the system (nuclear option):"
    echo "sudo reboot"
    
else
    echo "✅ Port 1883 is now FREE!"
    
    echo
    echo "Fix 5: Try starting mosquitto..."
    sudo systemctl start mosquitto
    
    sleep 3
    
    if systemctl is-active mosquitto >/dev/null 2>&1; then
        echo "✅ Mosquitto started successfully!"
        
        echo "📊 Mosquitto is now using port 1883:"
        sudo netstat -tulpn | grep :1883 || sudo ss -tulpn | grep :1883
        
        echo
        echo "📋 Recent logs:"
        tail -3 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo "No mosquitto.log available"
        
    else
        echo "❌ Mosquitto failed to start even with port free"
        echo "📋 Error details:"
        sudo systemctl status mosquitto --no-pager -l | tail -10
    fi
fi

echo
echo "6. 📊 FINAL STATUS"
echo "================="

echo "🔌 Port 1883 status:"
if sudo netstat -tulpn | grep :1883 >/dev/null; then
    echo "✅ Port 1883 is in use by:"
    sudo netstat -tulpn | grep :1883
else
    echo "❌ Port 1883 is not in use"
fi

echo
echo "🚀 Mosquitto service status:"
if systemctl is-active mosquitto >/dev/null 2>&1; then
    echo "✅ Mosquitto is RUNNING"
else
    echo "❌ Mosquitto is NOT RUNNING"
    echo "📋 Service status:"
    sudo systemctl status mosquitto --no-pager -l | head -5
fi

echo
echo "🧪 Quick test (if mosquitto is running):"
if systemctl is-active mosquitto >/dev/null 2>&1; then
    if timeout 5 mosquitto_pub -h localhost -p 1883 -u tenant1_admin -P admin123 -t "test/diagnostic" -m "test" 2>/dev/null; then
        echo "✅ MQTT test successful!"
    else
        echo "❌ MQTT test failed - check authentication"
    fi
else
    echo "⚠️  Cannot test - mosquitto not running"
fi

echo
echo "🎯 SUMMARY & NEXT STEPS"
echo "======================"
echo "1. Check what process was using port 1883 above"
echo "2. If port is now free and mosquitto running - you're good!"
echo "3. If port still in use - manually kill the process shown above"
echo "4. If mosquitto fails to start with port free - check config file"
echo "5. Consider clean reinstall only if configuration is corrupted"