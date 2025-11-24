/**
 * Real-Time CAN Bus Intrusion Detection System (Multiclass)
 * Uses decision tree extracted from neural network via Trustee
 * 
 * Detects 5 types of attacks:
 * - 0: Normal
 * - 1: DoS
 * - 2: Fuzzy
 * - 3: Gear_Attack  
 * - 4: RPM_Attack
 * 
 * Compile:
 *   g++ -std=c++17 -O3 -o live_ids live_ids_multiclass.cpp
 * 
 * Run:
 *   ./live_ids <can_log_file>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <map>

// Configuration
const int WINDOW_SIZE = 29;
const int NUM_FEATURES = 10;  // CAN_ID, DLC, DATA[0-7]
const int NUM_CLASSES = 5;

// Class names
const char* CLASS_NAMES[] = {
    "Normal",
    "DoS",
    "Fuzzy",
    "Gear_Attack",
    "RPM_Attack"
};

// Statistics tracking
struct Statistics {
    int total_messages = 0;
    int normal = 0;
    int dos = 0;
    int fuzzy = 0;
    int gear_attack = 0;
    int rpm_attack = 0;
    
    void update(int prediction) {
        total_messages++;
        switch(prediction) {
            case 0: normal++; break;
            case 1: dos++; break;
            case 2: fuzzy++; break;
            case 3: gear_attack++; break;
            case 4: rpm_attack++; break;
        }
    }
    
    void print() const {
        std::cout << "\n=== Detection Statistics ===" << std::endl;
        std::cout << "Total messages: " << total_messages << std::endl;
        std::cout << "  Normal:      " << normal << " (" << (100.0*normal/total_messages) << "%)" << std::endl;
        std::cout << "  DoS:         " << dos << " (" << (100.0*dos/total_messages) << "%)" << std::endl;
        std::cout << "  Fuzzy:       " << fuzzy << " (" << (100.0*fuzzy/total_messages) << "%)" << std::endl;
        std::cout << "  Gear_Attack: " << gear_attack << " (" << (100.0*gear_attack/total_messages) << "%)" << std::endl;
        std::cout << "  RPM_Attack:  " << rpm_attack << " (" << (100.0*rpm_attack/total_messages) << "%)" << std::endl;
    }
};

// CAN message structure
struct CANMessage {
    double timestamp;
    uint16_t can_id;
    uint8_t dlc;
    uint8_t data[8];
    
    std::vector<float> to_features() const {
        std::vector<float> features(NUM_FEATURES);
        features[0] = static_cast<float>(can_id);
        features[1] = static_cast<float>(dlc);
        for (int i = 0; i < 8; i++) {
            features[i+2] = static_cast<float>(data[i]);
        }
        return features;
    }
};

// Scaler (StandardScaler from Python)
class StandardScaler {
private:
    std::vector<float> mean_;
    std::vector<float> std_;
    
public:
    StandardScaler(const std::vector<float>& mean, const std::vector<float>& std)
        : mean_(mean), std_(std) {}
    
    std::vector<float> transform(const std::vector<float>& features) const {
        std::vector<float> scaled(features.size());
        for (size_t i = 0; i < features.size(); i++) {
            scaled[i] = (features[i] - mean_[i]) / std_[i];
        }
        return scaled;
    }
};

// Decision Tree Node (simplified implementation)
// In production, load tree structure from exported file
class SimpleDecisionTree {
public:
    int predict(const std::vector<float>& features) {
       
        
        // Simple example rules (replace with actual tree)
        float t0_can_id = features[0];    // Time 0, CAN_ID
        float t0_d0 = features[2];        // Time 0, DATA[0]
        float avg_can_id = 0;
        
        // Calculate average CAN_ID across window
        for (int t = 0; t < WINDOW_SIZE; t++) {
            avg_can_id += features[t * NUM_FEATURES + 0];
        }
        avg_can_id /= WINDOW_SIZE;
        
        // Simple classification rules (EXAMPLE ONLY)
        if (avg_can_id < 100) {
            return 1; // DoS (abnormally low IDs)
        } else if (avg_can_id > 1500) {
            return 3; // Gear_Attack (high IDs)
        } else if (t0_d0 > 200) {
            return 2; // Fuzzy (random high values)
        } else if (features[10] > features[20]) {  // Some pattern
            return 4; // RPM_Attack
        } else {
            return 0; // Normal
        }
    }
};

// Real-time detector with sliding window
class RealtimeDetector {
private:
    std::deque<CANMessage> window_;
    StandardScaler scaler_;
    SimpleDecisionTree tree_;
    Statistics stats_;
    int alert_threshold_;
    std::map<int, int> recent_alerts_;  // class -> count
    
public:
    RealtimeDetector(const StandardScaler& scaler, int alert_threshold = 5)
        : scaler_(scaler), alert_threshold_(alert_threshold) {}
    
    void process_message(const CANMessage& msg) {
        // Add to window
        window_.push_back(msg);
        
        // Keep window size fixed
        if (window_.size() > WINDOW_SIZE) {
            window_.pop_front();
        }
        
        // Need full window to classify
        if (window_.size() < WINDOW_SIZE) {
            return;
        }
        
        // Extract features from window
        std::vector<float> features;
        features.reserve(WINDOW_SIZE * NUM_FEATURES);
        
        for (const auto& m : window_) {
            auto msg_features = m.to_features();
            features.insert(features.end(), msg_features.begin(), msg_features.end());
        }
        
        // Scale features
        auto scaled_features = scaler_.transform(features);
        
        // Classify
        int prediction = tree_.predict(scaled_features);
        
        // Update statistics
        stats_.update(prediction);
        
        // Check for alerts
        if (prediction != 0) {  // Non-normal
            recent_alerts_[prediction]++;
            
            if (recent_alerts_[prediction] >= alert_threshold_) {
                raise_alert(prediction, msg);
                recent_alerts_[prediction] = 0;  // Reset after alert
            }
        } else {
            recent_alerts_.clear();  // Reset if normal message seen
        }
    }
    
    void raise_alert(int attack_class, const CANMessage& msg) {
        std::cout << "\n🚨 ALERT: " << CLASS_NAMES[attack_class] << " detected!" << std::endl;
        std::cout << "  Timestamp: " << std::fixed << std::setprecision(2) << msg.timestamp << std::endl;
        std::cout << "  CAN ID: 0x" << std::hex << msg.can_id << std::dec << std::endl;
        std::cout << "  Data: ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)msg.data[i] << " ";
        }
        std::cout << std::dec << std::endl;
    }
    
    void print_statistics() const {
        stats_.print();
    }
};

// Parse CAN message from log file
bool parse_can_message(const std::string& line, CANMessage& msg) {
    std::istringstream iss(line);
    std::string token;
    
    // Try to parse different formats
    // Format 1: "Timestamp: X ID: Y R DLC: Z data..."
    if (line.find("Timestamp:") != std::string::npos) {
        iss >> token >> msg.timestamp;  // "Timestamp:" value
        iss >> token >> token;          // "ID:" value (hex)
        msg.can_id = std::stoi(token, nullptr, 16);
        iss >> token;                   // "R" flag
        iss >> token >> token;          // "DLC:" value
        msg.dlc = std::stoi(token);
        
        for (int i = 0; i < 8; i++) {
            iss >> token;
            msg.data[i] = std::stoi(token, nullptr, 16);
        }
        return true;
    }
    
    // Format 2: CSV "timestamp,can_id,dlc,d0,d1,..."
    std::vector<std::string> tokens;
    while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() >= 11) {
        msg.timestamp = std::stod(tokens[0]);
        msg.can_id = std::stoi(tokens[1], nullptr, 0);  // Auto-detect hex/dec
        msg.dlc = std::stoi(tokens[2]);
        for (int i = 0; i < 8; i++) {
            msg.data[i] = std::stoi(tokens[i+3], nullptr, 0);
        }
        return true;
    }
    
    return false;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Real-Time CAN Bus Multiclass IDS ===" << std::endl;
    std::cout << "Detecting: Normal, DoS, Fuzzy, Gear Attack, RPM Attack" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <can_log_file>" << std::endl;
        return 1;
    }
    
    // Load scaler parameters (from Python export)
    // TODO: Load these from file generated by trustee_multiclass.py
    std::vector<float> scaler_mean(NUM_FEATURES, 0.0f);
    std::vector<float> scaler_std(NUM_FEATURES, 1.0f);
    
    // Example values (replace with actual values from trained model)
    scaler_mean = {695.97, 8.0, 70.54, 51.35, 42.49, 71.32, 58.62, 65.55, 29.05, 52.95};
    scaler_std  = {370.10, 0.01, 95.98, 59.66, 62.14, 94.54, 86.66, 77.20, 60.16, 71.25};
    
    StandardScaler scaler(scaler_mean, scaler_std);
    
    // Create detector
    RealtimeDetector detector(scaler, 5);  // Alert after 5 consecutive detections
    
    // Open CAN log file
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
        return 1;
    }
    
    std::cout << "\nProcessing CAN messages from: " << argv[1] << std::endl;
    std::cout << "Press Ctrl+C to stop...\n" << std::endl;
    
    // Process messages
    std::string line;
    int line_count = 0;
    int processed = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (std::getline(infile, line)) {
        line_count++;
        
        // Skip empty lines and headers
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        CANMessage msg;
        if (parse_can_message(line, msg)) {
            detector.process_message(msg);
            processed++;
            
            // Progress indicator
            if (processed % 10000 == 0) {
                std::cout << "\rProcessed: " << processed << " messages" << std::flush;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n\nProcessing complete!" << std::endl;
    std::cout << "Total lines: " << line_count << std::endl;
    std::cout << "Processed messages: " << processed << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "Throughput: " << (processed * 1000.0 / duration.count()) << " messages/sec" << std::endl;
    
    detector.print_statistics();
    
    return 0;
}

