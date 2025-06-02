#!/usr/bin/env python3
"""
MQTT Subscriber Script - Run on AWS Lightsail
This subscribes to ALL test topics and shows which messages actually get delivered
"""

import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import threading

# MQTT Configuration
MQTT_HOST = "localhost"  # Local on AWS
MQTT_PORT = 1883

# Subscriber credentials (use a user that can read all topics for monitoring)
SUBSCRIBER_USER = "tenant1_admin"  # Has broad access to see all
SUBSCRIBER_PASS = "admin123"

# Test topics to monitor
MONITOR_TOPICS = [
    "iot/tenant_001/+/+",      # Tenant 1 topics
    "iot/tenant_002/+/+",      # Tenant 2 topics  
    "admin/tenant_001/+",      # Admin topics
    "admin/tenant_002/+",      # Admin topics
    "test/+/+",               # Test topics
    "#"                       # Catch-all for any other topics
]

class MQTTSubscriber:
    def __init__(self):
        self.messages_received = []
        self.connection_status = "Disconnected"
        self.start_time = datetime.now()
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connection_status = "Connected"
            print(f"✅ Connected to MQTT broker as {SUBSCRIBER_USER}")
            print(f"📡 Subscribing to monitor topics...")
            
            # Subscribe to all monitor topics
            for topic in MONITOR_TOPICS:
                client.subscribe(topic, qos=1)
                print(f"   📋 Subscribed to: {topic}")
                
            print(f"\n🔍 MONITORING ACTIVE - Waiting for messages...")
            print(f"{'='*80}")
            print(f"{'Timestamp':<20} | {'Topic':<40} | {'Message'}")
            print(f"{'='*80}")
        else:
            self.connection_status = f"Failed (code: {rc})"
            print(f"❌ Connection failed: {rc}")
    
    def on_message(self, client, userdata, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        topic = message.topic
        payload = message.payload.decode('utf-8', errors='ignore')
        
        # Store message
        msg_data = {
            'timestamp': timestamp,
            'topic': topic,
            'payload': payload,
            'qos': message.qos
        }
        self.messages_received.append(msg_data)
        
        # Real-time display
        payload_short = payload[:30] + "..." if len(payload) > 30 else payload
        print(f"{timestamp:<20} | {topic:<40} | {payload_short}")
        
        # Analyze the message
        self.analyze_message(topic, payload)
        
    def analyze_message(self, topic, payload):
        """Analyze if this message should have been allowed based on ACL rules"""
        try:
            # Try to parse as JSON to get test metadata
            data = json.loads(payload)
            
            if 'test_id' in data and 'expected_delivery' in data:
                expected = data['expected_delivery']
                test_id = data['test_id']
                
                if expected:
                    print(f"   ✅ EXPECTED: Test {test_id} - Message correctly delivered")
                else:
                    print(f"   🚨 UNEXPECTED: Test {test_id} - Message should have been blocked by ACL!")
                    
        except (json.JSONDecodeError, KeyError):
            # Regular message, just note it was received
            print(f"   📨 Regular message received")
    
    def on_disconnect(self, client, userdata, rc):
        self.connection_status = "Disconnected"
        print(f"🔌 Disconnected from broker (code: {rc})")
    
    def on_log(self, client, userdata, level, buf):
        if level <= mqtt.MQTT_LOG_WARNING:
            print(f"📝 Log: {buf}")
    
    def print_status(self):
        """Print periodic status updates"""
        while True:
            time.sleep(30)  # Status every 30 seconds
            elapsed = datetime.now() - self.start_time
            print(f"\n📊 STATUS UPDATE (Running for {elapsed}):")
            print(f"   Connection: {self.connection_status}")
            print(f"   Messages received: {len(self.messages_received)}")
            print(f"   Monitoring topics: {len(MONITOR_TOPICS)}")
            print(f"{'='*80}")
    
    def run(self):
        """Start the subscriber"""
        print(f"🔍 MQTT MESSAGE DELIVERY MONITOR")
        print(f"{'='*50}")
        print(f"📡 Broker: {MQTT_HOST}:{MQTT_PORT}")
        print(f"👤 Subscriber: {SUBSCRIBER_USER}")
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"")
        
        # Create MQTT client
        client = mqtt.Client(
            client_id=f"acl_monitor_{int(time.time())}", 
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
        
        # Set credentials
        client.username_pw_set(SUBSCRIBER_USER, SUBSCRIBER_PASS)
        
        # Set callbacks
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.on_disconnect = self.on_disconnect
        client.on_log = self.on_log
        
        # Start status thread
        status_thread = threading.Thread(target=self.print_status)
        status_thread.daemon = True
        status_thread.start()
        
        try:
            # Connect and start monitoring
            print(f"🔗 Connecting to broker...")
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            
            # Keep monitoring
            client.loop_forever()
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            self.print_summary()
        except Exception as e:
            print(f"\n💥 Error: {e}")
        finally:
            client.disconnect()
    
    def print_summary(self):
        """Print summary of received messages"""
        print(f"\n📊 MONITORING SUMMARY")
        print(f"{'='*50}")
        print(f"⏰ Duration: {datetime.now() - self.start_time}")
        print(f"📨 Total messages received: {len(self.messages_received)}")
        
        if self.messages_received:
            print(f"\n📋 All received messages:")
            for i, msg in enumerate(self.messages_received, 1):
                print(f"{i:2d}. [{msg['timestamp']}] {msg['topic']} -> {msg['payload'][:50]}")
        else:
            print(f"❌ No messages received")

def main():
    subscriber = MQTTSubscriber()
    subscriber.run()

if __name__ == "__main__":
    print("🔍 MQTT ACL VERIFICATION - SUBSCRIBER MONITOR")
    print("=" * 60)
    print("This script monitors ALL topics to show which messages actually get delivered")
    print("Run this on AWS Lightsail, then run the publisher script on your Mac")
    print("=" * 60)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Subscriber stopped")
    except Exception as e:
        print(f"\n💥 Error: {e}")