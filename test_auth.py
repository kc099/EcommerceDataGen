#!/usr/bin/env python3
"""
MQTT Publisher Script - Run on Mac
This connects to AWS Lightsail and sends test messages to verify ACL blocking
"""

import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import sys

# MQTT Configuration - UPDATE WITH YOUR AWS LIGHTSAIL PUBLIC IP
MQTT_HOST = "13.232.105.192"  # Your AWS Lightsail public IP
MQTT_PORT = 1883

# Test scenarios with detailed analysis
TEST_SCENARIOS = [
    {
        "test_id": "001",
        "name": "Valid User + Valid Topic (Should Succeed)",
        "username": "tenant1_user1",
        "password": "user123",
        "topic": "iot/tenant_001/device_test/commands",
        "expected_delivery": True,
        "acl_rule": "tenant1_user1 has iot/tenant_001/+/commands (rw=2)",
        "reason": "User has write permission to iot/tenant_001/+/commands pattern"
    },
    {
        "test_id": "002", 
        "name": "Valid User + Cross-Tenant Topic (Should Fail)",
        "username": "tenant1_user1",
        "password": "user123",
        "topic": "iot/tenant_002/device_test/data",
        "expected_delivery": False,
        "acl_rule": "tenant1_user1 has NO access to tenant_002 topics",
        "reason": "Cross-tenant access should be blocked by ACL"
    },
    {
        "test_id": "003",
        "name": "Valid User + Admin Topic (Should Fail)",
        "username": "tenant1_user1", 
        "password": "user123",
        "topic": "admin/tenant_001/test_alerts",
        "expected_delivery": False,
        "acl_rule": "tenant1_user1 has NO access to admin/* topics",
        "reason": "Regular user should not access admin topics"
    },
    {
        "test_id": "004",
        "name": "Admin User + Valid Topic (Depends on Superuser Setting)",
        "username": "tenant1_admin",
        "password": "admin123", 
        "topic": "iot/tenant_001/admin_test/data",
        "expected_delivery": True,
        "acl_rule": "tenant1_admin has iot/tenant_001/+/+ (rw=3)",
        "reason": "Admin has readwrite access to tenant_001 topics"
    },
    {
        "test_id": "005",
        "name": "Admin User + Cross-Tenant Topic (Should Fail if Superuser Disabled)",
        "username": "tenant1_admin",
        "password": "admin123",
        "topic": "iot/tenant_002/admin_test/data", 
        "expected_delivery": False,
        "acl_rule": "tenant1_admin has NO access to tenant_002 topics",
        "reason": "Cross-tenant access should be blocked even for admins (if superuser disabled)"
    },
    {
        "test_id": "006",
        "name": "Admin User + Admin Topic (Should Succeed)",
        "username": "tenant1_admin",
        "password": "admin123",
        "topic": "admin/tenant_001/admin_test_alert",
        "expected_delivery": True, 
        "acl_rule": "tenant1_admin has admin/tenant_001/# (rw=3)",
        "reason": "Admin should have access to admin topics in their tenant"
    },
    {
        "test_id": "007",
        "name": "Invalid Credentials (Should Fail at Connection)",
        "username": "invalid_user",
        "password": "wrong_password",
        "topic": "test/invalid/credentials",
        "expected_delivery": False,
        "acl_rule": "N/A - Authentication should fail",
        "reason": "Invalid credentials should be rejected at connection level"
    }
]

class MQTTPublisher:
    def __init__(self):
        self.test_results = []
        self.current_test = None
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"    ✅ Connected successfully")
            self.current_test["connection_success"] = True
        else:
            print(f"    ❌ Connection failed (code: {rc})")
            self.current_test["connection_success"] = False
            self.current_test["connection_error"] = f"Connection failed with code {rc}"
    
    def on_publish(self, client, userdata, mid):
        print(f"    📤 Publish command sent (Message ID: {mid})")
        self.current_test["publish_sent"] = True
        
    def on_disconnect(self, client, userdata, rc):
        print(f"    🔌 Disconnected (code: {rc})")
    
    def on_log(self, client, userdata, level, buf):
        if level <= mqtt.MQTT_LOG_WARNING:
            print(f"    📝 Log: {buf}")
    
    def run_test_scenario(self, scenario):
        """Run a single test scenario"""
        print(f"\n{'='*100}")
        print(f"🧪 TEST {scenario['test_id']}: {scenario['name']}")
        print(f"{'='*100}")
        print(f"👤 User: {scenario['username']}")
        print(f"📍 Topic: {scenario['topic']}")
        print(f"🎯 Expected Delivery: {'YES' if scenario['expected_delivery'] else 'NO'}")
        print(f"🛡️  ACL Rule: {scenario['acl_rule']}")
        print(f"💡 Reason: {scenario['reason']}")
        print()
        
        # Initialize test result
        self.current_test = {
            "test_id": scenario["test_id"],
            "name": scenario["name"],
            "username": scenario["username"],
            "topic": scenario["topic"],
            "expected_delivery": scenario["expected_delivery"],
            "connection_success": False,
            "publish_sent": False,
            "connection_error": None
        }
        
        # Create test message with metadata
        test_message = {
            "test_id": scenario["test_id"],
            "test_name": scenario["name"],
            "expected_delivery": scenario["expected_delivery"],
            "timestamp": datetime.now().isoformat(),
            "username": scenario["username"],
            "acl_rule": scenario["acl_rule"],
            "reason": scenario["reason"],
            "data": f"Test message from {scenario['username']} at {datetime.now()}"
        }
        
        message_json = json.dumps(test_message, indent=2)
        
        # Create MQTT client
        client_id = f"test_publisher_{scenario['test_id']}_{int(time.time())}"
        client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        
        # Set credentials
        client.username_pw_set(scenario["username"], scenario["password"])
        
        # Set callbacks
        client.on_connect = self.on_connect
        client.on_publish = self.on_publish
        client.on_disconnect = self.on_disconnect
        client.on_log = self.on_log
        
        try:
            print(f"🔗 Attempting to connect to {MQTT_HOST}:{MQTT_PORT}...")
            
            # Connect to broker
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_start()
            
            # Wait for connection
            time.sleep(3)
            
            if self.current_test["connection_success"]:
                print(f"📤 Publishing test message...")
                print(f"📋 Message size: {len(message_json)} bytes")
                
                # Publish the test message
                result = client.publish(scenario["topic"], message_json, qos=1)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"    ✅ Publish command completed successfully")
                    print(f"    📊 Message sent to topic: {scenario['topic']}")
                    
                    if scenario["expected_delivery"]:
                        print(f"    🎯 EXPECTATION: Message should appear in subscriber monitor")
                    else:
                        print(f"    🎯 EXPECTATION: Message should NOT appear in subscriber monitor (ACL blocked)")
                else:
                    print(f"    ❌ Publish failed with code: {result.rc}")
                    self.current_test["publish_error"] = f"Publish failed with code {result.rc}"
                
                # Wait a moment for message to be processed
                time.sleep(2)
            else:
                print(f"    ❌ Cannot publish - connection failed")
                if scenario["expected_delivery"]:
                    print(f"    🎯 UNEXPECTED: Connection should have succeeded")
                else:
                    print(f"    🎯 EXPECTED: Connection failure (authentication blocked)")
            
            # Clean disconnect
            client.loop_stop()
            client.disconnect()
            
        except Exception as e:
            print(f"    💥 Exception: {e}")
            self.current_test["exception"] = str(e)
        
        # Store result
        self.test_results.append(self.current_test.copy())
        
        # Brief pause between tests
        print(f"\n⏸️  Waiting 5 seconds before next test...")
        time.sleep(5)
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print(f"🚀 MQTT ACL VERIFICATION - PUBLISHER")
        print(f"{'='*60}")
        print(f"📡 Target Broker: {MQTT_HOST}:{MQTT_PORT}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧪 Total Tests: {len(TEST_SCENARIOS)}")
        print()
        print(f"📋 IMPORTANT: Make sure the subscriber is running on AWS Lightsail!")
        print(f"📋 The subscriber will show which messages actually get delivered.")
        print()
        
        input("Press Enter to start testing...")
        
        # Run each test scenario
        for i, scenario in enumerate(TEST_SCENARIOS, 1):
            print(f"\n🔄 Running test {i}/{len(TEST_SCENARIOS)}...")
            self.run_test_scenario(scenario)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary and analysis instructions"""
        print(f"\n{'='*100}")
        print(f"📊 PUBLISHER TEST SUMMARY")
        print(f"{'='*100}")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧪 Tests Run: {len(self.test_results)}")
        
        # Categorize results
        connection_successes = sum(1 for r in self.test_results if r.get("connection_success", False))
        connection_failures = len(self.test_results) - connection_successes
        
        print(f"🔗 Connection Results:")
        print(f"   ✅ Successful connections: {connection_successes}")
        print(f"   ❌ Failed connections: {connection_failures}")
        
        print(f"\n📋 Detailed Test Results:")
        for result in self.test_results:
            status = "✅ CONNECTED" if result.get("connection_success", False) else "❌ FAILED"
            expected = "📨 SHOULD DELIVER" if result["expected_delivery"] else "🚫 SHOULD BLOCK"
            
            print(f"   {result['test_id']}. {status} | {expected} | {result['name']}")
            
            if not result.get("connection_success", False) and result.get("connection_error"):
                print(f"       💡 {result['connection_error']}")
        
        print(f"\n🔍 ANALYSIS INSTRUCTIONS:")
        print(f"{'='*60}")
        print(f"Now check the SUBSCRIBER MONITOR on AWS Lightsail:")
        print()
        print(f"✅ Messages that SHOULD appear (expected_delivery: true):")
        for result in self.test_results:
            if result["expected_delivery"] and result.get("connection_success", False):
                print(f"   📨 Test {result['test_id']}: {result['topic']}")
        
        print(f"\n❌ Messages that should NOT appear (ACL should block):")
        for result in self.test_results:
            if not result["expected_delivery"] and result.get("connection_success", False):
                print(f"   🚫 Test {result['test_id']}: {result['topic']}")
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   ✅ Only the 'SHOULD appear' messages show up in subscriber")
        print(f"   ✅ None of the 'should NOT appear' messages show up in subscriber")
        print(f"   ✅ Invalid credentials (Test 007) should fail to connect")
        
        print(f"\n📊 IF ACL IS WORKING CORRECTLY:")
        print(f"   📨 You'll see Tests 001, 004, 006 messages in subscriber") 
        print(f"   🚫 You WON'T see Tests 002, 003, 005 messages in subscriber")
        print(f"   ❌ Test 007 should fail to connect entirely")
        
        print(f"\n📝 Compare subscriber output with these expectations!")

def main():
    # Validate broker IP
    print(f"🔍 MQTT ACL VERIFICATION - PUBLISHER")
    print(f"{'='*60}")
    print(f"📡 Target Broker: {MQTT_HOST}:{MQTT_PORT}")
    print()
    
    # Confirm broker IP
    confirm = input(f"Is {MQTT_HOST} the correct AWS Lightsail public IP? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Please update MQTT_HOST in the script with your correct AWS public IP")
        sys.exit(1)
    
    # Run tests
    publisher = MQTTPublisher()
    publisher.run_all_tests()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Publisher stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")