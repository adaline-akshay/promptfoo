import Groq from 'groq-sdk';
import type {
  CompleteChatHandlerResponseType,
} from '@adaline/gateway';
import { Groq as GatewayGroq } from '@adaline/groq';

import { getCache, isCacheEnabled } from '../cache';
import logger from '../logger';
import type {
  ApiProvider,
  CallApiContextParams,
  CallApiOptionsParams,
  EnvOverrides,
  ProviderResponse,
} from '../types';
import { maybeLoadFromExternalFile, renderVarsInObject } from '../util';
import { REQUEST_TIMEOUT_MS, parseChatPrompt } from './shared';

interface GroqCompletionOptions {
  apiKey?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: Array<{
    type: 'function';
    function: {
      name: string;
      description?: string;
      parameters?: Record<string, any>;
    };
  }>;
  tool_choice?: 'none' | 'auto' | { type: 'function'; function: { name: string } };
  functionToolCallbacks?: Record<string, (arg: string) => Promise<string>>;
  systemPrompt?: string;
}

export class GroqProvider implements ApiProvider {
  private groq: Groq;
  private modelName: string;
  private apiKey: string | undefined;
  public config: GroqCompletionOptions;

  constructor(
    modelName: string,
    options: { config?: GroqCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    const { config, env } = options;
    this.modelName = modelName;
    this.config = config || {};
    this.apiKey = this.config.apiKey || env?.GROQ_API_KEY || process.env.GROQ_API_KEY;
    this.groq = new Groq({
      apiKey: this.apiKey,
      maxRetries: 2,
      timeout: REQUEST_TIMEOUT_MS,
    });
  }

  id = () => `groq:${this.modelName}`;

  public getModelName(): string {
    return this.modelName;
  }

  toString(): string {
    return `[Groq Provider ${this.modelName}]`;
  }

  // TODO: remove all console.logs
  // TODO: add all logger.debug
  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      {
        role: 'system',
        content: this.config.systemPrompt || 'You are a helpful assistant.',
      },
      ...parseChatPrompt(prompt, [{ role: 'user' as const, content: prompt }]),
    ];

    const params = {
      messages,
      model: this.config.model || this.modelName,
      temperature: this.config.temperature ?? 0.7,
      max_tokens: this.config.max_tokens ?? 1000,
      top_p: this.config.top_p ?? 1,
      tools: this.config.tools ? maybeLoadFromExternalFile(this.config.tools) : undefined,
      tool_choice: this.config.tool_choice ?? 'auto',
    };

    if (context?.vars && this.config.tools) {
      params.tools = maybeLoadFromExternalFile(renderVarsInObject(this.config.tools, context.vars));
    }

    let chatCompletion,
      useGateway = context?.gateway ? true : false;

    if (useGateway) {
      try {
        console.log('Calling Azure Chat Completion via Gateway');
        const { data: gatewayData } = await this.callApiGateway(params, context);
        chatCompletion = gatewayData as any;
        console.log('chatCompletion', chatCompletion);
      } catch (err) {
        console.log('Error calling Azure Chat Completion via Gateway: ', err);
        logger.debug(`Error calling Azure Chat Completion via Gateway: ${err}`);
        useGateway = false;
      }
    }

    const cacheKey = `groq:${JSON.stringify(params)}`;
    if (!useGateway) {
      if (isCacheEnabled()) {
        const cachedResult = await getCache().get<ProviderResponse>(cacheKey);
        if (cachedResult) {
          logger.debug(`Returning cached response for ${prompt}: ${JSON.stringify(cachedResult)}`);
          return {
            ...cachedResult,
            tokenUsage: {
              ...cachedResult.tokenUsage,
              cached: cachedResult.tokenUsage?.total,
            },
          };
        }
      }

      try {
        chatCompletion = await this.groq.chat.completions.create(params);

        if (!chatCompletion?.choices?.[0]) {
          throw new Error('Invalid response from Groq API');
        }
      } catch (err: any) {
        logger.error(`Groq API call error: ${err}`);
        const errorMessage = err.status ? `${err.status} ${err.name}: ${err.message}` : `${err}`;
        return { error: `API call error: ${errorMessage}` };
      }
    }

    try {
      const { message } = chatCompletion.choices[0];
      let output = message.content || '';

      if (message.tool_calls?.length) {
        const toolCalls = message.tool_calls.map((toolCall: any) => ({
          id: toolCall.id,
          type: toolCall.type,
          function: {
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          },
        }));
        output = JSON.stringify(toolCalls);

        // Handle function tool callbacks
        if (this.config.functionToolCallbacks) {
          for (const toolCall of message.tool_calls) {
            if (toolCall.function && this.config.functionToolCallbacks[toolCall.function.name]) {
              const functionResult = await this.config.functionToolCallbacks[
                toolCall.function.name
              ](toolCall.function.arguments);
              output += `\n\n[Function Result: ${functionResult}]`;
            }
          }
        }
      }

      const result: ProviderResponse = {
        output,
        tokenUsage: {
          total: chatCompletion.usage?.total_tokens,
          prompt: chatCompletion.usage?.prompt_tokens,
          completion: chatCompletion.usage?.completion_tokens,
        },
      };

      if (isCacheEnabled() && !useGateway) {
        try {
          await getCache().set(cacheKey, result);
        } catch (err) {
          logger.error(`Failed to cache response: ${String(err)}`);
        }
      }

      return result;
    } catch (err: any) {
      logger.error(`Groq API call error: ${err}`);
      const errorMessage = err.status ? `${err.status} ${err.name}: ${err.message}` : `${err}`;
      return { error: `API call error: ${errorMessage}` };
    }
  }

  async callApiGateway(
    body: any,
    context?: CallApiContextParams,
  ): Promise<{ data: ProviderResponse; cached: boolean }> {
    try {
      const gatewayGroq = new GatewayGroq();
      // TODO: include config.model 
      if (!gatewayGroq.chatModelLiterals().includes(this.modelName)) {
        throw new Error(`Unsupported Gateway Groq chat model: ${this.modelName}`);
      }

      const gatewayModel = gatewayGroq.chatModel({
        modelName: this.modelName,
        apiKey: this.apiKey as string,
      });

      const gatewayRequest = gatewayModel.transformModelRequest(body);
      // TODO: maybe add this to the top of the method and all other updated methods
      logger.debug(`Calling Groq chat completion via Gateway: ${JSON.stringify(gatewayRequest)}`);
      const response = (await context?.gateway?.completeChat({
        model: gatewayModel,
        config: gatewayRequest.config,
        messages: gatewayRequest.messages,
        tools: gatewayRequest.tools,
        options: {
          enableCache: isCacheEnabled(),
        },
      })) as CompleteChatHandlerResponseType;
      logger.debug(`Groq chat completion via Gateway response: ${JSON.stringify(response)}`);
      const gatewayResponse = response.provider.response.data;
      return { data: gatewayResponse, cached: response.cached };
    } catch (err) {
      logger.debug(`Error calling Groq chat completion via Gateway: ${err}`);
      throw new Error(`Error calling Groq chat completion via Gateway: ${err}`);
    }
  }
}
