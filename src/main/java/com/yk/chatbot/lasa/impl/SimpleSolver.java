package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.lasa.Solve;
import com.yk.chatbot.lasa.SolutionResult;
import org.springframework.stereotype.Component;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * Solve 인터페이스의 기본 구현체입니다.
 * 분석된 의도에 따라 적절한 응답 전략을 수립합니다.
 */
@Component
public class SimpleSolver implements Solve {

    // 의도별 응답 템플릿
    private static final Map<String, String> responseTemplates = new HashMap<>();
    
    static {
        responseTemplates.put("greeting", "안녕하세요! 무엇을 도와드릴까요?");
        responseTemplates.put("weather", "현재 {{location}}의 날씨는 {{weather_status}}입니다. 기온은 {{temperature}}도 입니다.");
        responseTemplates.put("time_query", "현재 시간은 {{current_time}}입니다.");
        responseTemplates.put("help", "제가 도울 수 있는 것들은 다음과 같습니다: 날씨 정보 제공, 현재 시간 안내, 인사하기 등이 있습니다.");
        responseTemplates.put("unknown", "죄송합니다. 이해하지 못했어요. 다른 방식으로 질문해 주시겠어요?");
    }

    @Override
    public SolutionResult solve(AnalysisResult result) {
        if (result == null) {
            return createErrorSolution();
        }
        
        String intent = result.getIntent();
        Map<String, Object> data = new HashMap<>();
        
        switch (intent) {
            case "greeting":
                // 간단한 인사 의도는 추가 처리 없음
                return createSuccessSolution(intent, responseTemplates.get(intent), data);
                
            case "weather":
                // 날씨 정보 조회
                String location = result.getEntities().getOrDefault("location", "서울");
                data.put("location", location);
                data.put("weather_status", getWeatherForLocation(location));
                data.put("temperature", getTemperatureForLocation(location));
                return createSuccessSolution(intent, responseTemplates.get(intent), data);
                
            case "time_query":
                // 현재 시간 정보 제공
                String timePeriod = result.getEntities().getOrDefault("time_period", "");
                LocalTime now = LocalTime.now();
                String formattedTime = now.format(DateTimeFormatter.ofPattern("HH시 mm분 ss초"));
                data.put("current_time", formattedTime);
                return createSuccessSolution(intent, responseTemplates.get(intent), data);
                
            case "help":
                // 도움말 제공
                return createSuccessSolution(intent, responseTemplates.get(intent), data);
                
            default:
                // 알 수 없는 의도
                return createSuccessSolution("unknown", responseTemplates.get("unknown"), data);
        }
    }
    
    /**
     * 성공적인 해결 결과 객체를 생성합니다.
     */
    private SolutionResult createSuccessSolution(String intent, String template, Map<String, Object> data) {
        return SolutionResult.builder()
                .status("success")
                .originalIntent(intent)
                .responseTemplate(template)
                .data(data)
                .build();
    }
    
    /**
     * 오류 상태의 해결 결과 객체를 생성합니다.
     */
    private SolutionResult createErrorSolution() {
        return SolutionResult.builder()
                .status("error")
                .originalIntent("unknown")
                .responseTemplate("시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                .build();
    }
    
    /**
     * 지정된 위치의 날씨 정보를 가져옵니다.
     * 실제 구현에서는 외부 API를 호출하거나 데이터베이스 조회 등을 수행합니다.
     */
    private String getWeatherForLocation(String location) {
        // 실제 구현에서는 날씨 API 호출
        return "맑음";
    }
    
    /**
     * 지정된 위치의 온도를 가져옵니다.
     * 실제 구현에서는 외부 API를 호출하거나 데이터베이스 조회 등을 수행합니다.
     */
    private int getTemperatureForLocation(String location) {
        // 실제 구현에서는 날씨 API 호출
        return 23;
    }
}
