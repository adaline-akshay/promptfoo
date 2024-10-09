import type {
  CompleteChatHandlerResponseType,
} from '@adaline/gateway';
import { Google as GatewayGoogle } from '@adaline/google';

import { isCacheEnabled, fetchWithCache } from '../cache';
import { getEnvString } from '../envars';
import logger from '../logger';
import type { ApiProvider, EnvOverrides, ProviderResponse, CallApiContextParams } from '../types';
import { parseChatPrompt, REQUEST_TIMEOUT_MS } from './shared';

const DEFAULT_API_HOST = 'generativelanguage.googleapis.com';

interface PalmCompletionOptions {
  apiKey?: string;
  apiHost?: string;
  safetySettings?: { category: string; probability: string }[];
  stopSequences?: string[];
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  topK?: number;
  generationConfig?: Record<string, any>;
}

class PalmGenericProvider implements ApiProvider {
  modelName: string;

  config: PalmCompletionOptions;
  env?: EnvOverrides;

  constructor(
    modelName: string,
    options: { config?: PalmCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    const { config, id, env } = options;
    this.env = env;
    this.modelName = modelName;
    this.config = config || {};
    this.id = id ? () => id : this.id;
  }

  id(): string {
    return `palm:${this.modelName}`;
  }

  toString(): string {
    return `[Google AI Studio Provider ${this.modelName}]`;
  }

  getApiHost(): string | undefined {
    return (
      this.config.apiHost ||
      this.env?.GOOGLE_API_HOST ||
      this.env?.PALM_API_HOST ||
      getEnvString('GOOGLE_API_HOST') ||
      getEnvString('PALM_API_HOST') ||
      DEFAULT_API_HOST
    );
  }

  getApiKey(): string | undefined {
    return (
      this.config.apiKey ||
      this.env?.GOOGLE_API_KEY ||
      this.env?.PALM_API_KEY ||
      getEnvString('GOOGLE_API_KEY') ||
      getEnvString('PALM_API_KEY')
    );
  }

  // @ts-ignore: Prompt is not used in this implementation
  async callApi(prompt: string): Promise<ProviderResponse> {
    throw new Error('Not implemented');
  }
}

export class PalmChatProvider extends PalmGenericProvider {
  static CHAT_MODELS = ['chat-bison-001', 'gemini-pro', 'gemini-pro-vision'];

  constructor(
    modelName: string,
    options: { config?: PalmCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    if (!PalmChatProvider.CHAT_MODELS.includes(modelName)) {
      logger.warn(`Using unknown Google chat model: ${modelName}`);
    }
    super(modelName, options);
  }

  async callApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'Google API key is not set. Set the GOOGLE_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const isGemini = this.modelName.startsWith('gemini');
    if (isGemini) {
      return this.callGemini(prompt, context);
    }

    // https://developers.generativeai.google/tutorials/curl_quickstart
    // https://ai.google.dev/api/rest/v1beta/models/generateMessage
    const messages = parseChatPrompt(prompt, [{ content: prompt }]);
    const body = {
      prompt: { messages },
      temperature: this.config.temperature,
      topP: this.config.topP,
      topK: this.config.topK,

      safetySettings: this.config.safetySettings,
      stopSequences: this.config.stopSequences,
      maxOutputTokens: this.config.maxOutputTokens,
    };
    logger.debug(`Calling Google API: ${JSON.stringify(body)}`);

    let data;
    try {
      ({ data } = (await fetchWithCache(
        `https://${this.getApiHost()}/v1beta3/models/${
          this.modelName
        }:generateMessage?key=${this.getApiKey()}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      )) as unknown as any);
    } catch (err) {
      return {
        error: `API call error: ${String(err)}`,
      };
    }

    logger.debug(`\tGoogle API response: ${JSON.stringify(data)}`);

    if (!data?.candidates || data.candidates.length === 0) {
      return {
        error: `API did not return any candidate responses: ${JSON.stringify(data)}`,
      };
    }

    try {
      const output = data.candidates[0].content;
      return {
        output,
      };
    } catch (err) {
      return {
        error: `API response error: ${String(err)}: ${JSON.stringify(data)}`,
      };
    }
  }

  // TODO: remove all console.logs
  // TODO: add all logger.debug
  async callGemini(
    prompt: string,
    context?: CallApiContextParams,
  ): Promise<ProviderResponse> {
    const contents = parseChatPrompt(prompt, [{ parts: [{ text: prompt }] }]);
    const body = {
      contents,
      generationConfig: {
        temperature: this.config.temperature,
        topP: this.config.topP,
        topK: this.config.topK,
        stopSequences: this.config.stopSequences,
        maxOutputTokens: this.config.maxOutputTokens,
        ...this.config.generationConfig,
      },
      safetySettings: this.config.safetySettings,
    };
    logger.debug(`Calling Google API: ${JSON.stringify(body)}`);

    let data,
      useGateway = context?.gateway ? true : false;

    if (useGateway) {
      try {
        console.log('Calling Google AI Studio Chat Completion via Gateway');
        const { data: gatewayData } = await this.callApiGateway(body, context);
        data = gatewayData as any;
        console.log('data', data);
      } catch (err) {
        console.log('Error calling Google AI Studio Chat Completion via Gateway: ', err);
        logger.debug(`Error calling Google AI Studio Chat Completion via Gateway: ${err}`);
        useGateway = false;
      }
    }

    if (!useGateway) {
      try {
        // https://ai.google.dev/docs/gemini_api_overview#curl
        // https://ai.google.dev/tutorials/rest_quickstart
        ({ data } = (await fetchWithCache(
          `https://${this.getApiHost()}/v1beta/models/${
            this.modelName
          }:generateContent?key=${this.getApiKey()}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          },
          REQUEST_TIMEOUT_MS,
        )) as {
          data: {
            candidates: Array<{
              content: { parts: Array<{ text: string }> };
              safetyRatings: Array<{ category: string; probability: string }>;
            }>;
            promptFeedback?: { safetyRatings: Array<{ category: string; probability: string }> };
          };
        });
      } catch (err) {
        return {
          error: `API call error: ${String(err)}`,
        };
      }
    }

    logger.debug(`\tGoogle API response: ${JSON.stringify(data)}`);

    if (!data?.candidates || data.candidates.length === 0) {
      return {
        error: `API did not return any candidate responses: ${JSON.stringify(data)}`,
      };
    }

    const candidate = data.candidates[0];
    const parts = candidate.content.parts.map((part: { text: string }) => part.text).join('');

    try {
      return {
        output: parts,
      };
    } catch (err) {
      return {
        error: `API response error: ${String(err)}: ${JSON.stringify(data)}`,
      };
    }
  }

  async callApiGateway(
    body: any,
    context?: CallApiContextParams,
  ): Promise<{ data: ProviderResponse; cached: boolean }> {
    try {
      const gatewayGoogle = new GatewayGoogle();
      if (!gatewayGoogle.chatModelLiterals().includes(this.modelName)) {
        throw new Error(`Unsupported Gateway Google AI Studio chat model: ${this.modelName}`);
      }

      const gatewayModel = gatewayGoogle.chatModel({
        modelName: this.modelName,
        apiKey: this.getApiKey() as string,
      });

      // Adaline Gateway always expects role in prompts
      let lastRole = 'model';
      const transformedContents = body.contents.map((c: { parts: Array<{ text: string }> }) => {
        const role = lastRole === 'model' ? 'user' : 'model';
        lastRole = role;
        return {
          role,
          parts: c.parts,
        };
      });
      body.contents = transformedContents;

      const gatewayRequest = gatewayModel.transformModelRequest(body);
      const response = (await context?.gateway?.completeChat({
        model: gatewayModel,
        config: gatewayRequest.config,
        messages: gatewayRequest.messages,
        tools: gatewayRequest.tools,
        options: {
          enableCache: isCacheEnabled(),
        },
      })) as CompleteChatHandlerResponseType;
      logger.debug(`Google AI Studio chat completion via Gateway response: ${JSON.stringify(response)}`);
      const gatewayResponse = response.provider.response.data;
      return { data: gatewayResponse, cached: response.cached };
    } catch (err) {
      logger.debug(`Error calling Google AI Studio chat completion via Gateway: ${err}`);
      throw new Error(`Error calling Google AI Studio chat completion via Gateway: ${err}`);
    }
  }
}
