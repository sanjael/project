import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 60000,
})

export const predictionService = {
  async predict(file) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await api.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },

  async getReport(predictionData) {
    const { data } = await api.post('/predict/report', predictionData, {
      responseType: 'blob',
    })
    return data
  },
}

export default api
