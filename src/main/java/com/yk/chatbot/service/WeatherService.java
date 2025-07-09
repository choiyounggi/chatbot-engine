package com.yk.chatbot.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * 날씨 정보를 제공하는 서비스
 * OpenWeatherMap API를 사용하거나, 테스트 시에는 Mock 데이터를 제공합니다.
 */
@Slf4j
@Service
public class WeatherService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${openweathermap.api.key:dummy-api-key-for-tests}")
    private String apiKey;
    
    private final Map<String, String> cityCoordinates = new HashMap<>();
    
    /**
     * 생성자
     */
    public WeatherService(RestTemplateBuilder restTemplateBuilder, ObjectMapper objectMapper) {
        this.restTemplate = restTemplateBuilder.build();
        this.objectMapper = objectMapper;
        
        // 한국 주요 도시 좌표 정보 (위도, 경도)
        cityCoordinates.put("서울", "37.5665,126.9780");
        cityCoordinates.put("부산", "35.1796,129.0756");
        cityCoordinates.put("인천", "37.4563,126.7052");
        cityCoordinates.put("대구", "35.8714,128.6014");
        cityCoordinates.put("대전", "36.3504,127.3845");
        cityCoordinates.put("광주", "35.1595,126.8526");
        cityCoordinates.put("울산", "35.5384,129.3114");
        cityCoordinates.put("세종", "36.4800,127.2890");
        cityCoordinates.put("제주", "33.4996,126.5312");
    }
    
    /**
     * 특정 위치의 날씨 상태를 반환합니다.
     * 
     * @param location 위치명
     * @return 날씨 상태 문자열
     */
    public String getWeatherForLocation(String location) {
        // 테스트용 모의 응답 반환 (API 키가 dummy일 경우)
        if ("dummy-api-key-for-tests".equals(apiKey)) {
            return "맑음";
        }
        
        try {
            String coordinates = cityCoordinates.getOrDefault(location, cityCoordinates.get("서울"));
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "https://api.openweathermap.org/data/2.5/weather?lat=%s&lon=%s&appid=%s&lang=kr",
                latLon[0], latLon[1], apiKey
            );
            
            JsonNode response = objectMapper.readTree(restTemplate.getForObject(url, String.class));
            String weatherStatus = response.path("weather").get(0).path("description").asText();
            
            return weatherStatus;
        } catch (Exception e) {
            log.error("날씨 정보를 가져오는 중 오류 발생", e);
            return "알 수 없음";
        }
    }
    
    /**
     * 특정 위치의 기온을 반환합니다.
     * 
     * @param location 위치명
     * @return 기온(섭씨)
     */
    public int getTemperatureForLocation(String location) {
        // 테스트용 모의 응답 반환 (API 키가 dummy일 경우)
        if ("dummy-api-key-for-tests".equals(apiKey)) {
            return 23;
        }
        
        try {
            String coordinates = cityCoordinates.getOrDefault(location, cityCoordinates.get("서울"));
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "https://api.openweathermap.org/data/2.5/weather?lat=%s&lon=%s&appid=%s&units=metric",
                latLon[0], latLon[1], apiKey
            );
            
            JsonNode response = objectMapper.readTree(restTemplate.getForObject(url, String.class));
            int temperature = response.path("main").path("temp").asInt();
            
            return temperature;
        } catch (Exception e) {
            log.error("기온 정보를 가져오는 중 오류 발생", e);
            return 0;
        }
    }
}
