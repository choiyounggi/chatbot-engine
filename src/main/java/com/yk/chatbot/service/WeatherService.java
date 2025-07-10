package com.yk.chatbot.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import jakarta.annotation.PostConstruct;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 날씨 정보를 제공하는 서비스
 * OpenWeatherMap API를 사용하여 실시간 날씨 데이터를 제공합니다.
 */
@Slf4j
@Service
public class WeatherService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${openweathermap.api.key}")
    private String apiKey;
    
    private final Map<String, String> cityCoordinates = new HashMap<>();
    
    // 한국어 날씨 상태 매핑 (OpenWeatherMap API의 영어 날씨 상태를 한국어로 변환)
    private static final Map<String, String> WEATHER_CODE_TO_KOREAN = new HashMap<>();
    
    static {
        // 맑음/구름 관련
        WEATHER_CODE_TO_KOREAN.put("clear sky", "맑음");
        WEATHER_CODE_TO_KOREAN.put("few clouds", "구름 조금");
        WEATHER_CODE_TO_KOREAN.put("scattered clouds", "구름 조금");
        WEATHER_CODE_TO_KOREAN.put("broken clouds", "구름많음");
        WEATHER_CODE_TO_KOREAN.put("overcast clouds", "흐림");
        
        // 비/눈 관련
        WEATHER_CODE_TO_KOREAN.put("light rain", "약한 비");
        WEATHER_CODE_TO_KOREAN.put("moderate rain", "비");
        WEATHER_CODE_TO_KOREAN.put("heavy intensity rain", "강한 비");
        WEATHER_CODE_TO_KOREAN.put("very heavy rain", "폭우");
        WEATHER_CODE_TO_KOREAN.put("extreme rain", "폭우");
        WEATHER_CODE_TO_KOREAN.put("freezing rain", "진눈깨비");
        WEATHER_CODE_TO_KOREAN.put("light intensity shower rain", "소나기");
        WEATHER_CODE_TO_KOREAN.put("shower rain", "소나기");
        WEATHER_CODE_TO_KOREAN.put("heavy intensity shower rain", "강한 소나기");
        WEATHER_CODE_TO_KOREAN.put("ragged shower rain", "소나기");
        
        // 눈
        WEATHER_CODE_TO_KOREAN.put("light snow", "약한 눈");
        WEATHER_CODE_TO_KOREAN.put("snow", "눈");
        WEATHER_CODE_TO_KOREAN.put("heavy snow", "폭설");
        WEATHER_CODE_TO_KOREAN.put("sleet", "진눈깨비");
        WEATHER_CODE_TO_KOREAN.put("light shower sleet", "약한 진눈깨비");
        WEATHER_CODE_TO_KOREAN.put("shower sleet", "진눈깨비");
        WEATHER_CODE_TO_KOREAN.put("light shower snow", "약한 눈");
        WEATHER_CODE_TO_KOREAN.put("shower snow", "눈");
        WEATHER_CODE_TO_KOREAN.put("heavy shower snow", "폭설");
        
        // 안개/특수 기상 현상
        WEATHER_CODE_TO_KOREAN.put("mist", "안개");
        WEATHER_CODE_TO_KOREAN.put("smoke", "연무");
        WEATHER_CODE_TO_KOREAN.put("haze", "연무");
        WEATHER_CODE_TO_KOREAN.put("sand/dust whirls", "황사");
        WEATHER_CODE_TO_KOREAN.put("fog", "안개");
        WEATHER_CODE_TO_KOREAN.put("sand", "황사");
        WEATHER_CODE_TO_KOREAN.put("dust", "황사");
        
        // 천둥번개
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with light rain", "약한 비를 동반한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with rain", "비를 동반한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with heavy rain", "폭우를 동반한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("light thunderstorm", "약한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm", "천둥번개");
        WEATHER_CODE_TO_KOREAN.put("heavy thunderstorm", "강한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("ragged thunderstorm", "천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with light drizzle", "약한 이슬비를 동반한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with drizzle", "이슬비를 동반한 천둥번개");
        WEATHER_CODE_TO_KOREAN.put("thunderstorm with heavy drizzle", "강한 이슬비를 동반한 천둥번개");
        
        // 이슬비
        WEATHER_CODE_TO_KOREAN.put("light intensity drizzle", "약한 이슬비");
        WEATHER_CODE_TO_KOREAN.put("drizzle", "이슬비");
        WEATHER_CODE_TO_KOREAN.put("heavy intensity drizzle", "강한 이슬비");
        WEATHER_CODE_TO_KOREAN.put("light intensity drizzle rain", "약한 이슬비");
        WEATHER_CODE_TO_KOREAN.put("drizzle rain", "이슬비");
        WEATHER_CODE_TO_KOREAN.put("heavy intensity drizzle rain", "강한 이슬비");
        WEATHER_CODE_TO_KOREAN.put("shower rain and drizzle", "소나기와 이슬비");
        WEATHER_CODE_TO_KOREAN.put("heavy shower rain and drizzle", "강한 소나기와 이슬비");
        WEATHER_CODE_TO_KOREAN.put("shower drizzle", "이슬비");
    }
    
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
        
        // 추가 주요 도시 및 지역
        cityCoordinates.put("수원", "37.2636,127.0286");
        cityCoordinates.put("성남", "37.4449,127.1388");
        cityCoordinates.put("안양", "37.3941,126.9570");
        cityCoordinates.put("고양", "37.6559,126.8351");
        cityCoordinates.put("용인", "37.2344,127.2011");
        cityCoordinates.put("청주", "36.6424,127.4890");
        cityCoordinates.put("천안", "36.8151,127.1135");
        cityCoordinates.put("전주", "35.8242,127.1480");
        cityCoordinates.put("포항", "36.0199,129.3415");
        cityCoordinates.put("창원", "35.2540,128.6395");
        cityCoordinates.put("김해", "35.2282,128.8812");
        cityCoordinates.put("평택", "36.9920,127.0887");
        cityCoordinates.put("강릉", "37.7556,128.8961");
        cityCoordinates.put("거제", "34.8806,128.6211");
        cityCoordinates.put("양산", "35.3350,129.0378");
    }
    
    /**
     * 빈 초기화 후 API 키 검증
     */
    @PostConstruct
    public void init() {
        log.info("WeatherService 초기화 완료. API 키: {}", maskApiKey(apiKey));
        
        // API 키 검증
        if (apiKey == null || apiKey.trim().isEmpty()) {
            log.warn("OpenWeatherMap API 키가 설정되지 않았습니다. 테스트 모드로 실행합니다.");
        } else if ("dummy-api-key-for-tests".equals(apiKey)) {
            log.warn("테스트용 더미 API 키가 사용되고 있습니다. 테스트 모드로 실행합니다.");
        } else {
            log.info("OpenWeatherMap API 키가 설정되었습니다. 실시간 날씨 정보를 제공합니다.");
        }
    }
    
    /**
     * API 키를 마스킹하여 로그에 출력
     */
    private String maskApiKey(String key) {
        if (key == null || key.length() < 8) {
            return "***";
        }
        return key.substring(0, 4) + "..." + key.substring(key.length() - 4);
    }
    
    /**
     * 특정 위치의 날씨 상태를 반환합니다.
     * 
     * @param location 위치명
     * @return 날씨 상태 문자열
     */
    public String getWeatherForLocation(String location) {
        log.info("날씨 정보 요청 위치: {}", location);
        
        // 테스트용 API 키가 아닌지 확인 (null 체크 추가)
        boolean isRealApiKey = apiKey != null && !apiKey.trim().isEmpty() && !"dummy-api-key-for-tests".equals(apiKey);
        
        // 테스트용 모의 응답 반환 (API 키가 dummy이거나 없는 경우)
        if (!isRealApiKey) {
            log.info("테스트 모드에서 날씨 정보 제공 (실제 API 호출 안함)");
            LocalDateTime now = LocalDateTime.now();
            int hour = now.getHour();
            
            // 시간에 따라 다양한 날씨 상태 반환 (테스트용)
            if (hour >= 6 && hour < 12) {
                return "맑음";
            } else if (hour >= 12 && hour < 18) {
                return "구름많음";
            } else if (hour >= 18 && hour < 21) {
                return "흐림";
            } else {
                return "맑음";
            }
        }
        
        try {
            log.debug("실시간 날씨 정보 요청 시작: {}", location);
            String coordinates = getCoordinatesForLocation(location);
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "https://api.openweathermap.org/data/2.5/weather?lat=%s&lon=%s&appid=%s&lang=en",
                latLon[0], latLon[1], apiKey
            );
            
            log.debug("OpenWeatherMap API 호출: {}", url.replace(apiKey, "API_KEY_HIDDEN"));
            
            String rawResponse = restTemplate.getForObject(url, String.class);
            log.debug("API 응답 수신: {}", rawResponse);
            
            JsonNode response = objectMapper.readTree(rawResponse);
            String weatherStatus = response.path("weather").get(0).path("description").asText();
            log.info("날씨 상태 (영어): {}", weatherStatus);
            
            // 영어 날씨 상태를 한국어로 변환
            String koreanWeather = WEATHER_CODE_TO_KOREAN.getOrDefault(weatherStatus.toLowerCase(), "알 수 없음");
            log.info("날씨 상태 (한국어): {}", koreanWeather);
            
            return koreanWeather;
        } catch (Exception e) {
            log.error("날씨 정보를 가져오는 중 오류 발생: {}", e.getMessage(), e);
            return "알 수 없음 (오류 발생)";
        }
    }
    
    /**
     * 특정 위치의 기온을 반환합니다.
     * 
     * @param location 위치명
     * @return 기온(섭씨)
     */
    public int getTemperatureForLocation(String location) {
        log.info("기온 정보 요청 위치: {}", location);
        
        // 테스트용 API 키가 아닌지 확인 (null 체크 추가)
        boolean isRealApiKey = apiKey != null && !apiKey.trim().isEmpty() && !"dummy-api-key-for-tests".equals(apiKey);
        
        // 테스트용 모의 응답 반환 (API 키가 dummy이거나 없는 경우)
        if (!isRealApiKey) {
            log.info("테스트 모드에서 기온 정보 제공 (실제 API 호출 안함)");
            LocalDateTime now = LocalDateTime.now();
            int month = now.getMonthValue();
            int hour = now.getHour();
            
            // 월과 시간에 따라 다양한 온도 반환 (테스트용)
            if (month >= 3 && month <= 5) { // 봄
                return 15 + (hour > 12 ? 5 : 0);
            } else if (month >= 6 && month <= 8) { // 여름
                return 25 + (hour > 12 ? 5 : 0);
            } else if (month >= 9 && month <= 11) { // 가을
                return 15 + (hour > 12 ? 3 : -2);
            } else { // 겨울
                return 0 + (hour > 12 ? 5 : -3);
            }
        }
        
        try {
            log.debug("실시간 기온 정보 요청 시작: {}", location);
            String coordinates = getCoordinatesForLocation(location);
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "https://api.openweathermap.org/data/2.5/weather?lat=%s&lon=%s&appid=%s&units=metric",
                latLon[0], latLon[1], apiKey
            );
            
            log.debug("OpenWeatherMap API 호출: {}", url.replace(apiKey, "API_KEY_HIDDEN"));
            
            String rawResponse = restTemplate.getForObject(url, String.class);
            log.debug("API 응답 수신: {}", rawResponse);
            
            JsonNode response = objectMapper.readTree(rawResponse);
            int temperature = response.path("main").path("temp").asInt();
            log.info("기온 정보: {}°C", temperature);
            
            return temperature;
        } catch (Exception e) {
            log.error("기온 정보를 가져오는 중 오류 발생: {}", e.getMessage(), e);
            return 0;
        }
    }
    
    /**
     * 위치 이름에 해당하는 좌표를 반환합니다.
     * 좌표 정보가 없는 경우 OpenWeatherMap Geocoding API로 검색합니다.
     * 
     * @param location 위치명
     * @return "위도,경도" 형식의 좌표 문자열
     */
    private String getCoordinatesForLocation(String location) {
        // 입력값 정규화 (앞뒤 공백 제거, 특수문자 처리)
        location = location.trim();
        
        // 특수한 위치 처리 (예: "여기", "우리 동네" 등)
        if (location.equals("여기") || location.equals("이곳") || 
            location.equals("우리 동네") || location.equals("우리동네")) {
            log.info("특수 위치 '{}' -> '서울'로 변환", location);
            location = "서울";
        }
        
        // "시" 또는 "도" 접미사가 없는 경우 처리
        if (!location.endsWith("시") && !location.endsWith("도") && 
            !location.endsWith("군") && !location.endsWith("구")) {
            
            // 주요 도시 이름인지 확인
            for (String city : cityCoordinates.keySet()) {
                if (city.startsWith(location)) {
                    log.info("위치명 '{}' -> '{}' 확장", location, city);
                    location = city;
                    break;
                }
            }
        }
        
        // 미리 정의된 좌표가 있으면 반환
        if (cityCoordinates.containsKey(location)) {
            log.debug("미리 정의된 좌표 사용: {} -> {}", location, cityCoordinates.get(location));
            return cityCoordinates.get(location);
        }
        
        // 테스트용 API 키가 아닌지 확인
        boolean isRealApiKey = apiKey != null && !apiKey.trim().isEmpty() && !"dummy-api-key-for-tests".equals(apiKey);
        
        // API 키가 dummy일 경우 서울 좌표 반환
        if (!isRealApiKey) {
            log.info("API 키가 유효하지 않아 기본 위치(서울)의 좌표를 사용합니다");
            return cityCoordinates.get("서울");
        }
        
        try {
            // OpenWeatherMap Geocoding API 사용하여 위치 검색
            String url = String.format(
                "https://api.openweathermap.org/geo/1.0/direct?q=%s,KR&limit=1&appid=%s",
                location, apiKey
            );
            
            log.debug("Geocoding API 호출: {}", url.replace(apiKey, "API_KEY_HIDDEN"));
            
            String rawResponse = restTemplate.getForObject(url, String.class);
            log.debug("Geocoding API 응답: {}", rawResponse);
            
            JsonNode response = objectMapper.readTree(rawResponse);
            if (response.isArray() && response.size() > 0) {
                double lat = response.get(0).path("lat").asDouble();
                double lon = response.get(0).path("lon").asDouble();
                String coordinates = lat + "," + lon;
                
                log.info("위치 '{}' 좌표 찾음: {}", location, coordinates);
                
                // 새로 찾은 좌표를 캐시에 추가
                cityCoordinates.put(location, coordinates);
                return coordinates;
            }
        } catch (Exception e) {
            log.warn("위치 좌표를 찾는 중 오류 발생: {}", location, e);
        }
        
        // 찾지 못한 경우 서울 좌표 반환
        log.info("'{}' 위치를 찾을 수 없어 기본 위치(서울)의 좌표를 사용합니다", location);
        return cityCoordinates.get("서울");
    }
    
    /**
     * 모든 지원 도시 목록을 반환합니다.
     * 
     * @return 지원 도시 목록
     */
    public String[] getSupportedCities() {
        return cityCoordinates.keySet().toArray(new String[0]);
    }
    
    /**
     * API 키가 유효한지 확인합니다.
     */
    public boolean hasValidApiKey() {
        return apiKey != null && !apiKey.trim().isEmpty() && !"dummy-api-key-for-tests".equals(apiKey);
    }
}
