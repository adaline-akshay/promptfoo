Here's a quick usage guide and documentation for using the OpenAI Provider:

# OpenAI Provider Usage Guide

The OpenAI Provider is a class that implements the `ProviderV1` interface, allowing you to easily work with OpenAI's chat and embedding models.

## Initialization

```typescript
import { OpenAI } from '@adaline/openai';

const openai = new OpenAI();
```

## Chat Models

### List available chat models

```typescript
const chatModels = openai.chatModelLiterals();
console.log(chatModels);
```

### Create a chat model instance

```typescript
const modelName = 'gpt-4-turbo-preview';
const model = openai.chatModel(modelName, {
  apiKey: 'your-api-key-here',
});
```

### Get schema for a specific chat model

```typescript
const schema = openai.chatModelSchema(modelName);
console.log(schema);
```

### Get schemas for all chat models

```typescript
const allSchemas = openai.chatModelSchemas();
console.log(allSchemas);
```

## Embedding Models

### List available embedding models

```typescript
const embeddingModels = openai.embeddingModelLiterals();
console.log(embeddingModels);
```

### Create an embedding model instance

```typescript
const embeddingModelName = 'text-embedding-ada-002';
const embeddingModel = openai.embeddingModel(embeddingModelName, {
  apiKey: 'your-api-key-here',
});
```

### Get schema for a specific embedding model

```typescript
const embeddingSchema = openai.embeddingModelSchema(embeddingModelName);
console.log(embeddingSchema);
```

### Get schemas for all embedding models

```typescript
const allEmbeddingSchemas = openai.embeddingModelSchemas();
console.log(allEmbeddingSchemas);
```

## Error Handling

The provider throws `ProviderError` when an invalid model name is used or other errors occur. Always wrap your code in try-catch blocks to handle these errors gracefully.

```typescript
try {
  const invalidModel = openai.chatModel('invalid-model-name', {});
} catch (error) {
  if (error instanceof ProviderError) {
    console.error('Provider Error:', error.message);
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Custom Base URL

You can specify a custom base URL for API requests by including it in the options:

```typescript
const model = openai.chatModel('gpt-4', {
  apiKey: 'your-api-key-here',
  baseUrl: 'https://your-custom-endpoint.com',
});
```

## Notes

- Always keep your API key secure and never expose it in client-side code.
- The provider supports various OpenAI models, including GPT-3.5, GPT-4, and their variants.
- Make sure to check OpenAI's documentation for the latest information on model capabilities and best practices.

This guide covers the basic usage of the OpenAI Provider. For more detailed information on using the chat and embedding models, refer to the specific model documentation and the `@adaline/gateway` package for making API calls.
