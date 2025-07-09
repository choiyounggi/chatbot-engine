package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.lasa.Answer;
import com.yk.chatbot.lasa.SolutionResult;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Answer 인터페이스의 기본 구현체입니다.
 * 해결 결과를 사용자가 이해할 수 있는 응답으로 변환합니다.
 */
@Component
public class SimpleAnswerer implements Answer {

    // 템플릿의 변수 패턴 ({{variable_name}})
    private static final Pattern TEMPLATE_PATTERN = Pattern.compile("\\{\\{([\\w_]+)\\}\\}");

    @Override
    public ChatResponse answer(SolutionResult result) {
        if (result == null) {
            return ChatResponse.builder()
                    .reply("시스템 오류가 발생했습니다.")
                    .intent("error")
                    .build();
        }
        
        // 템플릿에 데이터 적용
        String finalResponse = applyTemplateData(result.getResponseTemplate(), result.getData());
        
        // ChatResponse 객체 생성 및 반환
        return ChatResponse.builder()
                .reply(finalResponse)
                .intent(result.getOriginalIntent())
                .build();
    }
    
    /**
     * 템플릿에 데이터를 적용하여 최종 문자열을 생성합니다.
     * 템플릿 내의 {{variable}} 형태의 변수를 데이터로 치환합니다.
     *
     * @param template 응답 템플릿
     * @param data 치환될 데이터
     * @return 데이터가 적용된 최종 문자열
     */
    private String applyTemplateData(String template, Map<String, Object> data) {
        if (template == null || data == null) {
            return template;
        }
        
        StringBuffer result = new StringBuffer();
        Matcher matcher = TEMPLATE_PATTERN.matcher(template);
        
        while (matcher.find()) {
            String variable = matcher.group(1);
            Object value = data.getOrDefault(variable, "");
            matcher.appendReplacement(result, value.toString());
        }
        
        matcher.appendTail(result);
        return result.toString();
    }
}
