FROM node:20-slim
WORKDIR /app

# 複製套件清單並安裝
COPY package*.json ./
RUN npm install

# 複製原始碼
COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]