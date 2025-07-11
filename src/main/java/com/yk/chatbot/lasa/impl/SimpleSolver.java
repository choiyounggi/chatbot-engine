package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.lasa.Solve;
import com.yk.chatbot.lasa.SolutionResult;
import com.yk.chatbot.service.OpenAIService;
import com.yk.chatbot.service.WeatherService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
@RequiredArgsConstructor
public class SimpleSolver implements Solve {

    private final WeatherService weatherService;
    private final OpenAIService openAIService;
    
    // 다양한 인사말 응답 패턴 추가
    private static final List<String> GREETING_RESPONSES = Arrays.asList(
        "안녕하세요! 무엇을 도와드릴까요?",
        "반갑습니다! 오늘은 어떤 도움이 필요하신가요?",
        "안녕하세요! 좋은 하루 되고 계신가요?",
        "어서오세요! 무엇을 알려드릴까요?",
        "반가워요! 무엇이든 물어보세요."
    );
    
    // 다양한 작별 인사 응답 패턴 추가
    private static final List<String> BYE_RESPONSES = Arrays.asList(
        "안녕히 가세요! 좋은 하루 되세요!",
        "다음에 또 봐요! 좋은 시간 되세요.",
        "이용해 주셔서 감사합니다. 또 필요하시면 언제든 불러주세요!",
        "좋은 하루 보내세요! 다음에 또 만나요.",
        "안녕히 계세요! 또 뵙겠습니다."
    );
    
    // 더 상세한 날씨 응답 템플릿
    private static final Map<String, String> WEATHER_DESCRIPTIONS = Map.of(
        "맑음", "화창한 날씨입니다! 야외 활동하기 좋은 날이에요.",
        "구름많음", "구름이 조금 있지만 나쁘지 않은 날씨입니다.",
        "흐림", "하늘이 흐리네요. 우산을 챙기는 것이 좋을 수 있어요.",
        "비", "비가 내리고 있어요. 외출 시 우산을 꼭 챙기세요!",
        "소나기", "소나기가 내리고 있어요. 잠시 실내에서 대기하는 것이 좋겠습니다.",
        "눈", "눈이 내리고 있어요. 미끄러울 수 있으니 조심하세요!",
        "안개", "안개가 끼었네요. 운전 시 특히 주의하세요.",
        "천둥번개", "천둥번개가 치고 있어요. 야외 활동은 위험할 수 있습니다."
    );
    
    private static final Map<String, String> RESPONSE_TEMPLATES = Map.of(
        "greeting", "%s", // 랜덤 인사말로 대체됨
        "weather", "%s의 현재 날씨는 %s입니다. %s",
        "temperature", "%s의 현재 기온은 %d°C입니다. %s",
        "time", "현재 시간은 %s입니다.",
        "bye", "%s", // 랜덤 작별 인사로 대체됨
        "thanks", "천만에요! 더 필요한 것이 있으면 말씀해주세요.",
        "help", "저는 날씨, 시간 정보를 알려드리거나 간단한 대화가 가능해요. '서울 날씨 어때?'나 '지금 몇시야?' 같은 질문을 해보세요.",
        "error", "%s",
        "fallback", "%s"
    );
    
    // 위치 키워드 목록 - 메시지에서 위치 정보 추출 시 사용
    private static final String[] LOCATION_KEYWORDS = {
        // 특별시/광역시
        "서울", "부산", "인천", "대구", "광주", "대전", "울산", "세종",
        "서울특별시", "부산광역시", "인천광역시", "대구광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        
        // 도
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주도", "제주특별자치도",
        
        // 주요 도시
        "수원", "성남", "안양", "안산", "고양", "용인", "청주", "천안", "전주", "포항", "창원",
        "김해", "평택", "강릉", "원주", "춘천", "속초", "여수", "순천", "목포", "경주", "구미",
        "거제", "양산", "진주", "파주", "의정부", "남양주", "화성", "시흥", "광명", "하남", "군포",
        "오산", "이천", "안성", "광주", "파주", "김포", "구리", "여주", "양주", "동두천", "과천",
        "의왕", "포천", "양평", "동해", "태백", "삼척", "정선", "홍천", "횡성", "영월", "평창",
        "정읍", "남원", "김제", "익산", "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안",
        "나주", "광양", "담양", "곡성", "구례", "고흥", "보성", "화순", "장흥", "강진", "해남",
        "영암", "무안", "함평", "영광", "장성", "완도", "진도", "신안",
        "영덕", "울진", "문경", "예천", "안동", "영양", "영주", "봉화", "울릉", "의성", "청송", "영천",
        "경산", "청도", "고령", "성주", "칠곡", "김천", "군위", "사천", "밀양", "의령", "함안",
        "창녕", "고성", "남해", "하동", "산청", "함양", "거창", "합천", "통영"
    };
    
    /**
     * 사용자 메시지에서 위치 정보를 추출합니다
     * @param message 사용자 메시지
     * @return 추출된 위치명 또는 기본값 "서울"
     */
    private String extractLocationFromMessage(String message) {
        if (message == null || message.isEmpty()) {
            return "서울";
        }
        
        for (String keyword : LOCATION_KEYWORDS) {
            if (message.contains(keyword)) {
                log.info("메시지에서 위치 키워드 추출 성공: {}", keyword);
                return keyword;
            }
        }
        
        log.info("메시지에서 위치를 찾을 수 없어 기본값 사용");
        return "서울";
    }
    
    @Override
    public SolutionResult solve(AnalysisResult result) {
        if (result == null) {
            return createErrorResult();
        }
        
        try {
            String intent = result.getIntent();
            Map<String, String> entities = result.getEntities();
            String originalMessage = result.getOriginalMessage();
            
            log.info("의도 처리 시작: intent={}, confidence={}", intent, result.getConfidence());
            
            switch (intent) {
                case "greeting":
                    return createSuccessResult(intent)
                            .addData("greeting", getRandomGreeting());
                    
                case "weather":
                    // 엔티티에서 지역 정보 추출 (지역이 없으면 기본값 "서울" 사용)
                    String location = "서울";
                    if (entities != null && entities.containsKey("location")) {
                        String extractedLocation = entities.get("location");
                        if (!extractedLocation.isEmpty()) {
                            location = extractedLocation;
                            log.info("날씨 요청 지역 엔티티 추출 성공: {}", location);
                        } else {
                            log.warn("날씨 요청 지역 엔티티가 빈 문자열입니다. 원본 메시지: '{}'", result.getOriginalMessage());
                        }
                    } else {
                        log.warn("날씨 요청에서 지역 엔티티를 찾을 수 없습니다. 원본 메시지: '{}'", result.getOriginalMessage());
                        
                        // 메시지에서 직접 위치 키워드 찾기 (백업 방법)
                        location = extractLocationFromMessage(result.getOriginalMessage());
                    }
                    
                    log.info("날씨 정보 요청 처리: 최종 위치 = {}", location);
                    String weather = weatherService.getWeatherForLocation(location);
                    String weatherDetail = WEATHER_DESCRIPTIONS.getOrDefault(weather, 
                            "오늘은 " + weather + " 상태입니다.");
                    
                    String currentDateTime = LocalDateTime.now()
                            .format(DateTimeFormatter.ofPattern("yyyy년 MM월 dd일 HH시 mm분"));
                    
                    return createSuccessResult(intent)
                            .addData("location", location)
                            .addData("weather", weather)
                            .addData("weatherDetail", weatherDetail)
                            .addData("dateTime", currentDateTime);
                    
                case "temperature":
                    // 엔티티에서 지역 정보 추출 (지역이 없으면 기본값 "서울" 사용)
                    location = "서울";
                    if (entities != null && entities.containsKey("location")) {
                        String extractedLocation = entities.get("location");
                        if (!extractedLocation.isEmpty()) {
                            location = extractedLocation;
                            log.info("기온 요청 지역 엔티티 추출 성공: {}", location);
                        } else {
                            log.warn("기온 요청 지역 엔티티가 빈 문자열입니다. 원본 메시지: '{}'", result.getOriginalMessage());
                        }
                    } else {
                        log.warn("기온 요청에서 지역 엔티티를 찾을 수 없습니다. 원본 메시지: '{}'", result.getOriginalMessage());
                        
                        // 메시지에서 직접 위치 키워드 찾기 (백업 방법)
                        location = extractLocationFromMessage(result.getOriginalMessage());
                    }
                    
                    log.info("기온 정보 요청 처리: 최종 위치 = {}", location);
                    int temp = weatherService.getTemperatureForLocation(location);
                    
                    // 온도에 따른 추가 설명
                    String tempDescription = "";
                    if (temp <= 0) {
                        tempDescription = "매우 춥습니다. 따뜻하게 입으세요!";
                    } else if (temp <= 10) {
                        tempDescription = "쌀쌀합니다. 겉옷을 챙기세요.";
                    } else if (temp <= 20) {
                        tempDescription = "선선한 날씨입니다.";
                    } else if (temp <= 28) {
                        tempDescription = "따뜻한 날씨입니다.";
                    } else {
                        tempDescription = "더운 날씨입니다. 시원하게 지내세요!";
                    }
                    
                    return createSuccessResult(intent)
                            .addData("location", location)
                            .addData("temperature", temp)
                            .addData("tempDescription", tempDescription);
                    
                case "time":
                    String currentTime = LocalDateTime.now()
                            .format(DateTimeFormatter.ofPattern("a h시 mm분"));
                    return createSuccessResult(intent)
                            .addData("time", currentTime);
                    
                case "bye":
                    return createSuccessResult(intent)
                            .addData("bye", getRandomBye());
                    
                case "thanks":
                case "help":
                    return createSuccessResult(intent);
                    
                case "error":
                    // error 처리 - OpenAI 호출
                    SolutionResult errorResult = createErrorResult();
                    enrichWithOpenAIResponse(errorResult, originalMessage, "error");
                    return errorResult;
                    
                default:
                    // fallback 처리 - OpenAI 호출
                    SolutionResult fallbackResult = createFallbackResult();
                    enrichWithOpenAIResponse(fallbackResult, originalMessage, "fallback");
                    return fallbackResult;
            }
        } catch (Exception e) {
            log.error("의도 처리 중 오류 발생", e);
            return createErrorResult();
        }
    }
    
    /**
     * 랜덤한 인사말을 반환합니다.
     */
    private String getRandomGreeting() {
        return GREETING_RESPONSES.get(new Random().nextInt(GREETING_RESPONSES.size()));
    }
    
    /**
     * 랜덤한 작별 인사를 반환합니다.
     */
    private String getRandomBye() {
        return BYE_RESPONSES.get(new Random().nextInt(BYE_RESPONSES.size()));
    }
    
    private SolutionResult createSuccessResult(String intent) {
        return SolutionResult.builder()
                .status("success")
                .originalIntent(intent)
                .responseTemplate(RESPONSE_TEMPLATES.getOrDefault(intent, RESPONSE_TEMPLATES.get("fallback")))
                .build();
    }
    
    private SolutionResult createErrorResult() {
        return SolutionResult.builder()
                .status("error")
                .originalIntent("error")
                .responseTemplate("%s") // OpenAI 응답으로 대체할 것이므로 단순 포맷 사용
                .data(new HashMap<>())
                .build()
                .addData("fallbackResponse", "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.");
    }
    
    private SolutionResult createFallbackResult() {
        return SolutionResult.builder()
                .status("success")
                .originalIntent("fallback")
                .responseTemplate("%s") // OpenAI 응답으로 대체할 것이므로 단순 포맷 사용
                .data(new HashMap<>()) // 빈 데이터맵으로 초기화
                .build()
                .addData("fallbackResponse", "잘 이해하지 못했습니다. 도움이 필요하시면 '도움말'이라고 입력해주세요.");
    }
    
    /**
     * SolutionResult에 OpenAI 응답을 추가합니다.
     * 
     * @param result 결과 객체
     * @param userMessage 사용자 메시지
     * @param intentType 의도 타입 (fallback 또는 error)
     */
    private void enrichWithOpenAIResponse(SolutionResult result, String userMessage, String intentType) {
        try {
            String aiResponse = openAIService.generateResponseAsync(userMessage)
                    .get(5, TimeUnit.SECONDS); // 5초 타임아웃으로 응답 대기
            
            if (aiResponse != null && !aiResponse.isEmpty()) {
                log.info("OpenAI로 {} 응답 생성: {}", intentType, aiResponse);
                result.addData("fallbackResponse", aiResponse); // fallbackResponse 키를 공통으로 사용
            } else {
                log.warn("OpenAI 응답 생성 실패, 기본 {} 응답 사용", intentType);
            }
        } catch (Exception e) {
            log.error("OpenAI 응답 대기 중 시간 초과 또는 오류 발생", e);
        }
    }
}
