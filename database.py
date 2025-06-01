#!/usr/bin/env python3
"""
Multi-Tenant IoT Platform Database Setup for Mosquitto Go Auth
This script creates tables and populates them with sample data for a multi-tenant IoT platform
"""

import mysql.connector
import hashlib
import base64
import os
from datetime import datetime

# Database configuration - UPDATE THESE WITH YOUR ACTUAL CREDENTIALS
DB_CONFIG = {
    'host': '68.178.150.182',
    'port': 3306,
    'user': 'kc099',
    'password': 'Roboworks23!',
    'database': 'mosquittoauth'
}

def create_pbkdf2_hash(password, salt_size=16, iterations=100000):
    """
    Create PBKDF2 hash compatible with mosquitto-go-auth
    Format: PBKDF2$sha512$iterations$salt_base64$hash_base64
    """
    salt = os.urandom(salt_size)
    key = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, iterations, dklen=64)
    
    salt_b64 = base64.b64encode(salt).decode('ascii')
    key_b64 = base64.b64encode(key).decode('ascii')
    
    return f"PBKDF2$sha512${iterations}${salt_b64}${key_b64}"

def create_database_schema(cursor):
    """Create the database schema for multi-tenant IoT platform"""
    
    # Drop existing tables (be careful in production!)
    drop_tables = [
        "DROP TABLE IF EXISTS device_acls",
        "DROP TABLE IF EXISTS user_acls", 
        "DROP TABLE IF EXISTS devices",
        "DROP TABLE IF EXISTS users"
    ]
    
    for query in drop_tables:
        cursor.execute(query)
        print(f"Executed: {query}")
    
    # Create users table (tenants/customers)
    create_users_table = """
    CREATE TABLE users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) NOT NULL UNIQUE,
        password_hash VARCHAR(300) NOT NULL,
        email VARCHAR(150),
        is_admin BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        tenant_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_username (username),
        INDEX idx_tenant_id (tenant_id)
    )
    """
    cursor.execute(create_users_table)
    print("Created users table")
    
    # Create devices table
    create_devices_table = """
    CREATE TABLE devices (
        id INT AUTO_INCREMENT PRIMARY KEY,
        device_id VARCHAR(100) NOT NULL UNIQUE,
        device_name VARCHAR(150),
        user_id INT NOT NULL,
        tenant_id VARCHAR(50) NOT NULL,
        device_type VARCHAR(50),
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_device_id (device_id),
        INDEX idx_user_id (user_id),
        INDEX idx_tenant_id (tenant_id)
    )
    """
    cursor.execute(create_devices_table)
    print("Created devices table")
    
    # Create user_acls table (permissions for users)
    create_user_acls_table = """
    CREATE TABLE user_acls (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        topic_pattern VARCHAR(300) NOT NULL,
        access_type INT NOT NULL COMMENT '1=read, 2=write, 3=readwrite, 4=subscribe',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_user_id (user_id)
    )
    """
    cursor.execute(create_user_acls_table)
    print("Created user_acls table")
    
    # Create device_acls table (specific device permissions)
    create_device_acls_table = """
    CREATE TABLE device_acls (
        id INT AUTO_INCREMENT PRIMARY KEY,
        device_id INT NOT NULL,
        topic_pattern VARCHAR(300) NOT NULL,
        access_type INT NOT NULL COMMENT '1=read, 2=write, 3=readwrite, 4=subscribe',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
        INDEX idx_device_id (device_id)
    )
    """
    cursor.execute(create_device_acls_table)
    print("Created device_acls table")

def populate_sample_data(cursor):
    """Populate tables with sample multi-tenant data"""
    
    # Sample users (tenants)
    users_data = [
        {
            'username': 'tenant1_admin',
            'password': 'admin123',
            'email': 'admin@tenant1.com',
            'is_admin': True,
            'tenant_id': 'tenant_001'
        },
        {
            'username': 'tenant1_user1',
            'password': 'user123',
            'email': 'user1@tenant1.com',
            'is_admin': False,
            'tenant_id': 'tenant_001'
        },
        {
            'username': 'tenant2_admin',
            'password': 'admin456',
            'email': 'admin@tenant2.com',
            'is_admin': True,
            'tenant_id': 'tenant_002'
        },
        {
            'username': 'tenant2_user1',
            'password': 'user456',
            'email': 'user1@tenant2.com',
            'is_admin': False,
            'tenant_id': 'tenant_002'
        }
    ]
    
    print("Creating users...")
    user_ids = {}
    for user in users_data:
        password_hash = create_pbkdf2_hash(user['password'])
        
        insert_user = """
        INSERT INTO users (username, password_hash, email, is_admin, tenant_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_user, (
            user['username'], 
            password_hash, 
            user['email'], 
            user['is_admin'], 
            user['tenant_id']
        ))
        
        user_ids[user['username']] = cursor.lastrowid
        print(f"Created user: {user['username']} (ID: {cursor.lastrowid})")
    
    # Sample devices
    devices_data = [
        {
            'device_id': 'sensor_001_temp',
            'device_name': 'Temperature Sensor 1',
            'username': 'tenant1_admin',
            'tenant_id': 'tenant_001',
            'device_type': 'temperature_sensor'
        },
        {
            'device_id': 'sensor_001_humidity', 
            'device_name': 'Humidity Sensor 1',
            'username': 'tenant1_admin',
            'tenant_id': 'tenant_001',
            'device_type': 'humidity_sensor'
        },
        {
            'device_id': 'actuator_001_valve',
            'device_name': 'Water Valve 1',
            'username': 'tenant1_user1',
            'tenant_id': 'tenant_001', 
            'device_type': 'valve_actuator'
        },
        {
            'device_id': 'sensor_002_temp',
            'device_name': 'Temperature Sensor 2',
            'username': 'tenant2_admin',
            'tenant_id': 'tenant_002',
            'device_type': 'temperature_sensor'
        },
        {
            'device_id': 'actuator_002_led',
            'device_name': 'LED Strip 2',
            'username': 'tenant2_user1',
            'tenant_id': 'tenant_002',
            'device_type': 'led_actuator'
        }
    ]
    
    print("Creating devices...")
    device_ids = {}
    for device in devices_data:
        insert_device = """
        INSERT INTO devices (device_id, device_name, user_id, tenant_id, device_type)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_device, (
            device['device_id'],
            device['device_name'], 
            user_ids[device['username']],
            device['tenant_id'],
            device['device_type']
        ))
        
        device_ids[device['device_id']] = cursor.lastrowid
        print(f"Created device: {device['device_id']} (ID: {cursor.lastrowid})")
    
    # User ACLs (tenant-level permissions)
    user_acls_data = [
        # Tenant 1 Admin - full access to tenant 1 topics
        ('tenant1_admin', 'iot/tenant_001/+/+', 3),  # readwrite
        ('tenant1_admin', 'iot/tenant_001/+/+', 4),  # subscribe
        ('tenant1_admin', 'admin/tenant_001/#', 3),   # admin topics
        
        # Tenant 1 User - limited access
        ('tenant1_user1', 'iot/tenant_001/+/data', 1),    # read data
        ('tenant1_user1', 'iot/tenant_001/+/data', 4),    # subscribe to data
        ('tenant1_user1', 'iot/tenant_001/+/commands', 2), # write commands
        
        # Tenant 2 Admin - full access to tenant 2 topics
        ('tenant2_admin', 'iot/tenant_002/+/+', 3),  # readwrite
        ('tenant2_admin', 'iot/tenant_002/+/+', 4),  # subscribe
        ('tenant2_admin', 'admin/tenant_002/#', 3),   # admin topics
        
        # Tenant 2 User - limited access  
        ('tenant2_user1', 'iot/tenant_002/+/data', 1),    # read data
        ('tenant2_user1', 'iot/tenant_002/+/data', 4),    # subscribe to data
        ('tenant2_user1', 'iot/tenant_002/+/commands', 2), # write commands
    ]
    
    print("Creating user ACLs...")
    for username, topic_pattern, access_type in user_acls_data:
        insert_acl = """
        INSERT INTO user_acls (user_id, topic_pattern, access_type)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_acl, (user_ids[username], topic_pattern, access_type))
        print(f"Created ACL: {username} -> {topic_pattern} (access: {access_type})")

def create_views_for_mosquitto(cursor):
    """Create views that mosquitto-go-auth will use"""
    
    # View for user authentication (userquery)
    create_auth_view = """
    CREATE OR REPLACE VIEW mosquitto_users AS
    SELECT username, password_hash as password FROM users WHERE is_active = 1
    """
    cursor.execute(create_auth_view)
    print("Created mosquitto_users view")
    
    # View for superuser check (superquery) 
    create_superuser_view = """
    CREATE OR REPLACE VIEW mosquitto_superusers AS
    SELECT username, is_admin as is_superuser FROM users WHERE is_active = 1
    """
    cursor.execute(create_superuser_view)
    print("Created mosquitto_superusers view")
    
    # View for ACL check (aclquery)
    create_acl_view = """
    CREATE OR REPLACE VIEW mosquitto_acls AS
    SELECT 
        u.username,
        ua.topic_pattern as topic,
        ua.access_type as rw
    FROM users u
    JOIN user_acls ua ON u.id = ua.user_id
    WHERE u.is_active = 1
    
    UNION ALL
    
    SELECT 
        u.username,
        da.topic_pattern as topic,
        da.access_type as rw  
    FROM users u
    JOIN devices d ON u.id = d.user_id
    JOIN device_acls da ON d.id = da.device_id
    WHERE u.is_active = 1 AND d.is_active = 1
    """
    cursor.execute(create_acl_view)
    print("Created mosquitto_acls view")

def main():
    """Main function to set up the database"""
    
    print("=== Multi-Tenant IoT Database Setup ===")
    print("IMPORTANT: Update DB_CONFIG with your actual database credentials!")
    print()
    
    try:
        # Connect to database
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        print(f"Connected to MySQL database: {DB_CONFIG['database']}")
        
        # Create schema
        print("\n1. Creating database schema...")
        create_database_schema(cursor)
        
        # Populate sample data
        print("\n2. Populating sample data...")
        populate_sample_data(cursor)
        
        # Create views for mosquitto
        print("\n3. Creating views for mosquitto-go-auth...")
        create_views_for_mosquitto(cursor)
        
        # Commit changes
        connection.commit()
        
        print("\n=== Setup Complete! ===")
        print("\nSample Users Created:")
        print("- tenant1_admin (password: admin123) - Tenant 1 Administrator")
        print("- tenant1_user1 (password: user123) - Tenant 1 Regular User")  
        print("- tenant2_admin (password: admin456) - Tenant 2 Administrator")
        print("- tenant2_user1 (password: user456) - Tenant 2 Regular User")
        
        print("\nTopic Structure:")
        print("- iot/{tenant_id}/{device_id}/data - Device sensor data")
        print("- iot/{tenant_id}/{device_id}/commands - Device commands")
        print("- admin/{tenant_id}/# - Administrative topics")
        
        print("\nNext Steps:")
        print("1. Update your mosquitto.conf with the MySQL configuration")
        print("2. Test the authentication with mosquitto clients")
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    except Exception as err:
        print(f"Error: {err}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()