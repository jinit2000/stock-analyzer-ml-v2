pipeline {
    agent any

    environment {
        // Your Docker Hub repo
        DOCKER_IMAGE = "jinitpatel5/stock-analyzer-ml-v2"

        // Jenkins credentials ID you will create for Docker Hub
        DOCKER_CREDENTIALS_ID = "docker-hub-credentials"

        // Change to "python" if your Jenkins node uses that name
        PYTHON = "python3"
    }

    stages {
        stage('Checkout') {
            steps {
                // Pull code from Git (GitHub, etc.)
                checkout scm
            }
        }

        stage('Set up Python & Install deps') {
            steps {
                sh '''
                    ${PYTHON} -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Build dataset & train models') {
            steps {
                sh '''
                    . venv/bin/activate
                    ${PYTHON} -m scripts.build_dataset
                    ${PYTHON} -m scripts.train_models
                '''
            }
        }

        stage('Run tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                script {
                    def tag = "${env.BUILD_NUMBER}"   // e.g. 1, 2, 3,...
                    sh """
                        docker build -t ${DOCKER_IMAGE}:${tag} .
                        docker tag ${DOCKER_IMAGE}:${tag} ${DOCKER_IMAGE}:latest
                    """
                }
            }
        }

        stage('Push Docker image') {
            steps {
                script {
                    def tag = "${env.BUILD_NUMBER}"
                    withCredentials([usernamePassword(
                        credentialsId: DOCKER_CREDENTIALS_ID,
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh '''
                            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        '''
                        sh """
                            docker push ${DOCKER_IMAGE}:${tag}
                            docker push ${DOCKER_IMAGE}:latest
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}
