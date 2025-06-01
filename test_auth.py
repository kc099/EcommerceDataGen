#!/usr/bin/env python3
"""
MQTT Authentication Test Script
Tests the multi-tenant IoT authentication setup
"""

import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime

# MQTT Broker configuration
MQTT_HOST = "localhost"
MQTT_PORT = 1883

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Tenant 1 Admin - Valid Access",
        "username": "tenant1_admin",
        "password": "admin123",
        "topic": "iot/tenant_001/sensor_001_temp/data",
        "message": json.dumps({"temperature": 23.5, "timestamp": str(datetime.now())}),
        "should_succeed": True
    },
    {
        "name": "Tenant 1 Admin - Cross-tenant Access (Should Fail)",
        "username": "tenant1_admin", 
        "password": "admin123",
        "topic": "iot/tenant_002/sensor_002_temp/data",
        "message": json.dumps({"temperature": 25.0, "timestamp": str(datetime.now())}),
        "should_succeed": False
    },
    {
        "name": "Tenant 1 User - Limited Access to Commands",
        "username": "tenant1_user1",
        "password": "user123", 
        "topic": "iot/tenant_001/actuator_001_valve/commands",
        "message": json.dumps({"command": "open", "timestamp": str(datetime.now())}),
        "should_succeed": True
    },
    {
        "name": "Tenant 1 User - Invalid Topic Access (Should Fail)",
        "username": "tenant1_user1",
        "password": "user123",
        "topic": "admin/tenant_001/alerts",
        "message": json.dumps({"alert": "test", "timestamp": str(datetime.now())}),
        "should_succeed": False
    },
    {
        "name": "Invalid Credentials (Should Fail)",
        "username": "invalid_user",
        "password": "wrong_password",
        "topic": "iot/tenant_001/test/data",
        "message": json.dumps({"test": "data"}),
        "should_succeed": False
    }
]

class MQTTTester:
    def __init__(self):
        self.results = []
        self.current_test = None
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ Connected successfully (Result code: {rc})")
            self.current_test["connection_success"] = True
        else:
            print(f"❌ Connection failed (Result code: {rc})")
            self.current_test["connection_success"] = False
            self.current_test["error"] = f"Connection failed with code {rc}"
    
    def on_publish(self, client, userdata, mid):
        print(f"✅ Message published successfully (Message ID: {mid})")
        self.current_test["publish_success"] = True
        
    def on_disconnect(self, client, userdata, rc):
        print(f"🔌 Disconnected (Result code: {rc})")
        
    def on_log(self, client, userdata, level, buf):
        if level <= mqtt.MQTT_LOG_WARNING:
            print(f"📝 Log: {buf}")
    
    def test_scenario(self, scenario):
        """Test a single authentication scenario"""
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"Username: {scenario['username']}")
        print(f"Topic: {scenario['topic']}")
        print(f"Expected result: {'SUCCESS' if scenario['should_succeed'] else 'FAILURE'}")
        print(f"{'='*60}")
        
        # Initialize test result
        self.current_test = {
            "name": scenario["name"],
            "username": scenario["username"],
            "topic": scenario["topic"],
            "expected_success": scenario["should_succeed"],
            "connection_success": False,
            "publish_success": False,
            "error": None
        }
        
        # Create MQTT client
        client = mqtt.Client(client_id=f"test_client_{int(time.time())}")
        client.username_pw_set(scenario["username"], scenario["password"])
        
        # Set callbacks
        client.on_connect = self.on_connect
        client.on_publish = self.on_publish
        client.on_disconnect = self.on_disconnect
        client.on_log = self.on_log
        
        try:
            # Connect to broker
            print(f"🔗 Attempting to connect...")
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            
            # Start the loop to process callbacks
            client.loop_start()
            
            # Wait for connection
            time.sleep(2)
            
            if self.current_test["connection_success"]:
                # Attempt to publish
                print(f"📤 Publishing message...")
                result = client.publish(scenario["topic"], scenario["message"], qos=1)
                
                # Wait for publish
                time.sleep(2)
                
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    self.current_test["error"] = f"Publish failed with code {result.rc}"
                    print(f"❌ Publish failed: {result.rc}")
            
            # Clean disconnect
            client.loop_stop()
            client.disconnect()
            
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            self.current_test["error"] = str(e)
        
        # Evaluate result
        self.evaluate_test_result()
        self.results.append(self.current_test.copy())
        
        time.sleep(1)  # Brief pause between tests
    
    def evaluate_test_result(self):
        """Evaluate if the test result matches expectations"""
        expected = self.current_test["expected_success"]
        connection_ok = self.current_test["connection_success"]
        publish_ok = self.current_test.get("publish_success", False)
        
        if expected:
            # Should succeed
            if connection_ok and publish_ok:
                result = "✅ PASS"
                self.current_test["result"] = "PASS"
            else:
                result = "❌ FAIL (Expected success but got failure)"
                self.current_test["result"] = "FAIL"
        else:
            # Should fail
            if not connection_ok or not publish_ok:
                result = "✅ PASS (Correctly rejected)"
                self.current_test["result"] = "PASS"
            else:
                result = "❌ FAIL (Expected failure but got success)"
                self.current_test["result"] = "FAIL"
        
        print(f"\n🎯 Test Result: {result}")
        if self.current_test.get("error"):
            print(f"💡 Error details: {self.current_test['error']}")
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting MQTT Authentication Tests")
        print(f"📡 Broker: {MQTT_HOST}:{MQTT_PORT}")
        
        for scenario in TEST_SCENARIOS:
            self.test_scenario(scenario)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("📊 TEST SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for r in self.results if r["result"] == "PASS")
        failed = sum(1 for r in self.results if r["result"] == "FAIL")
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result["result"] == "PASS" else "❌"
            print(f"{i:2d}. {status} {result['name']}")
            if result["result"] == "FAIL" and result.get("error"):
                print(f"    💡 {result['error']}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! Your authentication setup is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check your configuration.")
            
        print("\n📝 Next steps:")
        print("1. Check /var/log/mosquitto/mosquitto.log for detailed broker logs")
        print("2. Check /var/log/mosquitto/auth.log for authentication logs")
        print("3. Verify database connectivity and query results")

def main():
    """Main function"""
    print("🔐 MQTT Multi-Tenant Authentication Tester")
    print("="*50)
    
    tester = MQTTTester()
    tester.run_all_tests()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")