// Barebones WiFi AP + HTTP page — NO atech SDK, NO Serial.
// Purpose: test the car's WiFi from ANY client (e.g. a phone browser) to decide
// whether the radio works at all. Brings up AP "carmin"/"minifactory" at
// 192.168.4.1. Open http://192.168.4.1 on a connected device -> "CAR WIFI OK".
// If a PHONE sees the page but the Mac can't reach the car, the Mac's networking
// is the problem (not the car). If the phone also can't, it's the car's radio.
#include <WiFi.h>

WiFiServer server(80);

void setup() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP("carmin", "minifactory");
  WiFi.setSleep(false);
  server.begin();
  server.setNoDelay(true);
}

void loop() {
  WiFiClient c = server.available();
  if (c) {
    unsigned long t = millis();
    while (c.connected() && millis() - t < 400) {
      if (c.available()) { c.read(); }
      else break;
    }
    c.print("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
            "<html><body style='font-family:sans-serif;font-size:8vw;text-align:center'>"
            "<h1>CAR WIFI OK</h1></body></html>");
    c.stop();
  }
  delay(10);
}
