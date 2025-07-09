package com.yk.chatbot.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Rasa NLU 서버와 통신하기 위한 클라이언트 서비스
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RasaClientService {

    private final ObjectMapper objectMapper;

    @Value("${rasa.server.url}")
    private String rasaServerUrl;

    @Value("${rasa.server.timeout:5000}")
    private int timeout;

    /**
     * 메시지를 Rasa 서버로 보내 의도와 엔티티를 추출합니다
     *
     * @param message 사용자 메시지
     * @param senderId 발신자 ID (선택적)
     * @return 의도, 엔티티, 신뢰도 점수를 포함한 분석 결과
     */
    public Map<String, Object> parseMessage(String message, String senderId) {
        try {
            Map<String, Object> result = new HashMap<>();

            // Rasa NLU 모델 분석 요청
            String modelEndpoint = rasaServerUrl + "/model/parse";
            Map<String, String> requestBody = new HashMap<>();
            requestBody.put("text", message);
            if (senderId != null && !senderId.isEmpty()) {
                requestBody.put("sender", senderId);
            }

            JsonNode response = sendRequest(modelEndpoint, requestBody);

            // 응답 파싱
            String intent = response.has("intent") && response.get("intent").has("name")
                    ? response.get("intent").get("name").asText()
                    : "unknown";
            
            double confidence = response.has("intent") && response.get("intent").has("confidence")
                    ? response.get("intent").get("confidence").asDouble()
                    : 0.0;

            Map<String, Object> entities = new HashMap<>();
            if (response.has("entities") && response.get("entities").isArray()) {
                response.get("entities").forEach(entity -> {
                    String entityName = entity.get("entity").asText();
                    String entityValue = entity.get("value").asText();
                    entities.put(entityName, entityValue);
                });
            }

            result.put("intent", intent);
            result.put("confidence", confidence);
            result.put("entities", entities);
            result.put("rawResponse", response);

            return result;
        } catch (Exception e) {
            log.error("Rasa 서버 통신 중 오류 발생", e);
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("intent", "error");
            errorResult.put("confidence", 0.0);
            errorResult.put("entities", new HashMap<>());
            errorResult.put("error", e.getMessage());
            return errorResult;
        }
    }

    /**
     * HTTP 요청을 Rasa 서버로 전송합니다
     */
    private JsonNode sendRequest(String url, Object body) throws IOException {
        try (CloseableHttpClient httpClient = createHttpClient()) {
            HttpPost httpPost = new HttpPost(url);
            httpPost.setEntity(new StringEntity(objectMapper.writeValueAsString(body), ContentType.APPLICATION_JSON));

            return httpClient.execute(httpPost, response -> {
                return objectMapper.readTree(response.getEntity().getContent());
            });
        }
    }

    /**
     * HTTP 클라이언트를 생성합니다
     */
    private CloseableHttpClient createHttpClient() {
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectTimeout(timeout, TimeUnit.MILLISECONDS)
                .setResponseTimeout(timeout, TimeUnit.MILLISECONDS)
                .build();

        return HttpClients.custom()
                .setDefaultRequestConfig(requestConfig)
                .build();
    }
}
