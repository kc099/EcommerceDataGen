import paho.mqtt.client as mqtt
import time

# --- Connection Details ---
broker_address = "13.203.107.174"
broker_port = 1883
username = "kc"
password = "9912364541"
topic = "lightsail/test"

# --- Callback Functions ---
def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker."""
    if rc == 0:
        print("✅ Connected to Mosquitto Broker!")
        # Subscribe to the topic upon successful connection
        client.subscribe(topic)
    else:
        print(f"❌ Failed to connect, return code {rc}\n")

def on_subscribe(client, userdata, mid, granted_qos):
    """Callback for when the client subscribes to a topic."""
    print(f"subscribed to topic: {topic} with QoS: {granted_qos[0]}")

def on_message(client, userdata, msg):
    """Callback for when a message is received from the broker."""
    print(f"Received message: '{str(msg.payload.decode('utf-8'))}' on topic '{msg.topic}'")

# --- Main Script ---
# Create an MQTT client instance
client = mqtt.Client(client_id="my_pc_client")

# Assign callback functions
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message

# Set username and password
client.username_pw_set(username, password)

# Connect to the broker
try:
    client.connect(broker_address, broker_port, 60)
except Exception as e:
    print(f"Error connecting to broker: {e}")
    exit()

# Start the network loop in a non-blocking way
client.loop_start()

# Publish a message after a short delay
print("\nPublishing message...")
time.sleep(2) # Wait for connection to establish
client.publish(topic, "Hello from my PC! 👋")
print("Message published!\n")

# Keep the script running to listen for messages
try:
    # Wait for the message to be received
    time.sleep(5)
except KeyboardInterrupt:
    print("Exiting...")

# Stop the loop and disconnect
client.loop_stop()
client.disconnect()
print("Disconnected.")