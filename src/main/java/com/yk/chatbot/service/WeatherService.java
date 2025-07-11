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
    
    @Value("${weather.api.key}")
    private String apiKey;
    
    @Value("${weather.api.url}")
    private String apiUrl;
    
    @Value("${weather.api.demo-mode:false}")
    private boolean demoMode;
    
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
        
        // 한국 특별시/광역시 (위도, 경도)
        cityCoordinates.put("서울", "37.5665,126.9780");
        cityCoordinates.put("부산", "35.1796,129.0756");
        cityCoordinates.put("인천", "37.4563,126.7052");
        cityCoordinates.put("대구", "35.8714,128.6014");
        cityCoordinates.put("대전", "36.3504,127.3845");
        cityCoordinates.put("광주", "35.1595,126.8526");
        cityCoordinates.put("울산", "35.5384,129.3114");
        cityCoordinates.put("세종", "36.4800,127.2890");
        
        // 제주 관련 지역 (관광지 포함)
        cityCoordinates.put("제주", "33.4996,126.5312");
        cityCoordinates.put("제주도", "33.4996,126.5312");
        cityCoordinates.put("서귀포", "33.2539,126.5602");
        cityCoordinates.put("한라산", "33.3617,126.5292");
        cityCoordinates.put("성산일출봉", "33.4587,126.9425");
        cityCoordinates.put("우도", "33.5219,126.9571");
        cityCoordinates.put("중문", "33.2444,126.4125");
        
        // 경기도 주요 도시
        cityCoordinates.put("수원", "37.2636,127.0286");
        cityCoordinates.put("성남", "37.4449,127.1388");
        cityCoordinates.put("안양", "37.3941,126.9570");
        cityCoordinates.put("고양", "37.6559,126.8351");
        cityCoordinates.put("용인", "37.2344,127.2011");
        cityCoordinates.put("부천", "37.5035,126.7664");
        cityCoordinates.put("안산", "37.3219,126.8309");
        cityCoordinates.put("남양주", "37.6360,127.2165");
        cityCoordinates.put("의정부", "37.7380,127.0337");
        cityCoordinates.put("평택", "36.9920,127.0887");
        cityCoordinates.put("시흥", "37.3798,126.8032");
        cityCoordinates.put("파주", "37.7599,126.7799");
        cityCoordinates.put("김포", "37.6151,126.7150");
        cityCoordinates.put("광명", "37.4791,126.8646");
        cityCoordinates.put("광주시", "37.4132,127.2547"); // 경기도 광주시
        cityCoordinates.put("하남", "37.5398,127.2054");
        
        // 강원도 주요 도시 및 관광지
        cityCoordinates.put("춘천", "37.8813,127.7300");
        cityCoordinates.put("원주", "37.3422,127.9201");
        cityCoordinates.put("강릉", "37.7556,128.8961");
        cityCoordinates.put("동해", "37.5248,129.1142");
        cityCoordinates.put("속초", "38.2071,128.5918");
        cityCoordinates.put("삼척", "37.4498,129.1674");
        cityCoordinates.put("태백", "37.1665,128.9886");
        cityCoordinates.put("설악산", "38.1193,128.4657");
        cityCoordinates.put("양양", "38.0754,128.6189");
        cityCoordinates.put("평창", "37.3706,128.3903");
        
        // 충청북도 주요 도시
        cityCoordinates.put("청주", "36.6424,127.4890");
        cityCoordinates.put("충주", "36.9912,127.9260");
        cityCoordinates.put("제천", "37.1324,128.1910");
        cityCoordinates.put("음성", "36.9400,127.6903");
        cityCoordinates.put("진천", "36.8556,127.4354");
        cityCoordinates.put("증평", "36.7850,127.5814");
        
        // 충청남도 주요 도시
        cityCoordinates.put("천안", "36.8151,127.1135");
        cityCoordinates.put("공주", "36.5633,127.2544");
        cityCoordinates.put("보령", "36.3494,126.6032");
        cityCoordinates.put("아산", "36.7897,127.0019");
        cityCoordinates.put("서산", "36.7850,126.4503");
        cityCoordinates.put("논산", "36.1893,127.1000");
        cityCoordinates.put("계룡", "36.2741,127.2506");
        cityCoordinates.put("당진", "36.8899,126.6461");
        cityCoordinates.put("태안", "36.7456,126.2980");
        
        // 전라북도 주요 도시 및 관광지
        cityCoordinates.put("전주", "35.8242,127.1480");
        cityCoordinates.put("군산", "35.9676,126.7366");
        cityCoordinates.put("익산", "35.9483,126.9577");
        cityCoordinates.put("정읍", "35.5700,126.8561");
        cityCoordinates.put("남원", "35.4164,127.3905");
        cityCoordinates.put("김제", "35.8033,126.8809");
        cityCoordinates.put("완주", "35.8441,127.1550");
        cityCoordinates.put("진안", "35.7913,127.4253");
        cityCoordinates.put("무주", "36.0068,127.6611");
        cityCoordinates.put("장수", "35.6471,127.5212");
        
        // 전라남도 주요 도시 및 관광지
        cityCoordinates.put("목포", "34.8118,126.3922");
        cityCoordinates.put("여수", "34.7604,127.6622");
        cityCoordinates.put("순천", "34.9506,127.4875");
        cityCoordinates.put("나주", "35.0158,126.7103");
        cityCoordinates.put("광양", "34.9400,127.6958");
        cityCoordinates.put("담양", "35.3220,126.9882");
        cityCoordinates.put("곡성", "35.2821,127.2923");
        cityCoordinates.put("구례", "35.2022,127.4628");
        cityCoordinates.put("보성", "34.7717,127.0799");
        cityCoordinates.put("화순", "35.0646,126.9857");
        cityCoordinates.put("해남", "34.5733,126.5989");
        cityCoordinates.put("영암", "34.8001,126.6968");
        cityCoordinates.put("무안", "34.9904,126.4816");
        
        // 경상북도 주요 도시 및 관광지
        cityCoordinates.put("포항", "36.0199,129.3415");
        cityCoordinates.put("경주", "35.8562,129.2246");
        cityCoordinates.put("김천", "36.1398,128.1135");
        cityCoordinates.put("안동", "36.5684,128.7294");
        cityCoordinates.put("구미", "36.1195,128.3444");
        cityCoordinates.put("영주", "36.8055,128.6240");
        cityCoordinates.put("영천", "35.9733,128.9388");
        cityCoordinates.put("상주", "36.4108,128.1592");
        cityCoordinates.put("문경", "36.5869,128.1873");
        cityCoordinates.put("경산", "35.8250,128.7415");
        
        // 경상남도 주요 도시 및 관광지
        cityCoordinates.put("창원", "35.2540,128.6395");
        cityCoordinates.put("진주", "35.1800,128.1108");
        cityCoordinates.put("통영", "34.8544,128.4332");
        cityCoordinates.put("사천", "35.0038,128.0641");
        cityCoordinates.put("김해", "35.2282,128.8812");
        cityCoordinates.put("밀양", "35.5042,128.7464");
        cityCoordinates.put("거제", "34.8806,128.6211");
        cityCoordinates.put("양산", "35.3350,129.0378");
        cityCoordinates.put("남해", "34.8376,127.8924");
        cityCoordinates.put("거창", "35.6868,127.9093");
        cityCoordinates.put("함안", "35.2723,128.4064");
        cityCoordinates.put("합천", "35.5671,128.1673");
        cityCoordinates.put("산청", "35.4156,127.8738");
        cityCoordinates.put("하동", "35.0674,127.7513");
        
        // 서울 주요 지역
        cityCoordinates.put("강남", "37.5172,127.0473");
        cityCoordinates.put("강북", "37.6396,127.0257");
        cityCoordinates.put("강서", "37.5509,126.8495");
        cityCoordinates.put("관악", "37.4784,126.9516");
        cityCoordinates.put("광진", "37.5384,127.0822");
        cityCoordinates.put("구로", "37.4954,126.8874");
        cityCoordinates.put("금천", "37.4566,126.8972");
        cityCoordinates.put("노원", "37.6543,127.0567");
        cityCoordinates.put("도봉", "37.6688,127.0470");
        cityCoordinates.put("동대문", "37.5741,127.0399");
        cityCoordinates.put("마포", "37.5637,126.9086");
        cityCoordinates.put("서대문", "37.5791,126.9368");
        cityCoordinates.put("송파", "37.5145,127.1060");
        cityCoordinates.put("종로", "37.5720,126.9793");
        cityCoordinates.put("중구", "37.5636,126.9975");
        
        // 인천 주요 지역
        cityCoordinates.put("부평", "37.5070,126.7218");
        cityCoordinates.put("남동", "37.4467,126.7312");
        cityCoordinates.put("서구", "37.5450,126.6756"); // 인천 서구
        cityCoordinates.put("연수", "37.4106,126.6780");
        cityCoordinates.put("계양", "37.5374,126.7380");
        cityCoordinates.put("강화", "37.7469,126.4881");
        
        // 주요 관광지 및 명소
        cityCoordinates.put("에버랜드", "37.2933,127.2025");
        cityCoordinates.put("롯데월드", "37.5111,127.0980");
        cityCoordinates.put("경복궁", "37.5796,126.9770");
        cityCoordinates.put("남산", "37.5514,126.9882");
        cityCoordinates.put("북한산", "37.6587,126.9773");
        cityCoordinates.put("독도", "37.2427,131.8675");
        cityCoordinates.put("울릉도", "37.5041,130.8667");
        cityCoordinates.put("지리산", "35.3349,127.7306");
        cityCoordinates.put("설악산", "38.1193,128.4657");
        cityCoordinates.put("월출산", "34.7547,126.6870");
        cityCoordinates.put("덕유산", "35.8606,127.7467");
        cityCoordinates.put("가야산", "35.8227,128.1147");
        cityCoordinates.put("소백산", "36.9059,128.4575");
        cityCoordinates.put("오대산", "37.7341,128.5986");
        cityCoordinates.put("내장산", "35.4977,126.8862");
        cityCoordinates.put("태안해안", "36.6733,126.2794");
        cityCoordinates.put("속리산", "36.5307,127.8572");
        cityCoordinates.put("월정리", "33.5561,126.7959"); // 제주 월정리
        cityCoordinates.put("해운대", "35.1586,129.1603"); // 부산 해운대
    }
    
    /**
     * 빈 초기화 후 API 키 검증
     */
    @PostConstruct
    public void init() {
        log.info("WeatherService 초기화 완료. API URL: {}, 데모 모드: {}", apiUrl, demoMode);
        
        // API 키 검증
        if (demoMode) {
            log.info("날씨 서비스가 데모 모드로 실행됩니다. 실제 API 호출 없이 가상 데이터를 제공합니다.");
        } else if (apiKey == null || apiKey.trim().isEmpty()) {
            log.warn("Weather API 키가 설정되지 않았습니다. 테스트 모드로 실행합니다.");
        } else if ("dummy-api-key-for-tests".equals(apiKey)) {
            log.warn("테스트용 더미 API 키가 사용되고 있습니다. 테스트 모드로 실행합니다.");
        } else {
            log.info("Weather API 키가 설정되었습니다: {}", maskApiKey(apiKey));
            log.info("실시간 날씨 정보를 제공합니다.");
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
        
        // 데모 모드이거나 API 키가 유효하지 않은 경우
        if (demoMode || !hasValidApiKey()) {
            log.info("데모/테스트 모드에서 날씨 정보 제공 (실제 API 호출 안함)");
            return generateDemoWeather();
        }
        
        try {
            log.debug("실시간 날씨 정보 요청 시작: {}", location);
            String coordinates = getCoordinatesForLocation(location);
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "%s?lat=%s&lon=%s&appid=%s&lang=en&units=metric",
                apiUrl, latLon[0], latLon[1], apiKey
            );
            
            log.debug("Weather API 호출: {}", url.replace(apiKey, "API_KEY_HIDDEN"));
            
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
     * 데모 모드용 가상 날씨 상태 생성
     * 
     * @return 랜덤 날씨 상태
     */
    private String generateDemoWeather() {
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
    
    /**
     * 특정 위치의 기온을 반환합니다.
     * 
     * @param location 위치명
     * @return 기온(섭씨)
     */
    public int getTemperatureForLocation(String location) {
        log.info("기온 정보 요청 위치: {}", location);
        
        // 데모 모드이거나 API 키가 유효하지 않은 경우
        if (demoMode || !hasValidApiKey()) {
            log.info("데모/테스트 모드에서 기온 정보 제공 (실제 API 호출 안함)");
            return generateDemoTemperature();
        }
        
        try {
            log.debug("실시간 기온 정보 요청 시작: {}", location);
            String coordinates = getCoordinatesForLocation(location);
            String[] latLon = coordinates.split(",");
            String url = String.format(
                "%s?lat=%s&lon=%s&appid=%s&units=metric",
                apiUrl, latLon[0], latLon[1], apiKey
            );
            
            log.debug("Weather API 호출: {}", url.replace(apiKey, "API_KEY_HIDDEN"));
            
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
     * 데모 모드용 가상 기온 데이터 생성
     * 
     * @return 계절과 시간에 맞는 기온
     */
    private int generateDemoTemperature() {
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
    
    /**
     * 도시 좌표 정보를 반환합니다.
     * @return 도시명과 좌표 정보(위도,경도)를 담은 Map
     */
    public Map<String, String> getCityCoordinates() {
        return cityCoordinates;
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
        
        // 데모 모드이거나 API 키가 유효하지 않은 경우 서울 좌표 반환
        if (demoMode || !hasValidApiKey()) {
            log.info("데모 모드이거나 API 키가 유효하지 않아 기본 위치(서울)의 좌표를 사용합니다");
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
