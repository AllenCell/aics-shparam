pipeline {
    parameters { booleanParam(name: 'create_release', defaultValue: false,
                              description: 'If true, create a release artifact and publish to ' +
                                           'the artifactory release PyPi or public PyPi.') }
    options {
        timeout(time: 1, unit: 'HOURS')
    }
    agent {
        node {
            label "python-gradle"
        }
    }
    environment {
        PATH = "/home/jenkins/.local/bin:$PATH"
        REQUESTS_CA_BUNDLE = "/etc/ssl/certs"
    }
    stages {
        stage ("create virtualenv") {
            steps {
                this.notifyAICSBitBucket("INPROGRESS")
                sh "./gradlew -i cleanAll installCIDependencies"
            }
        }

        stage ("bump version pre-build") {
            when {
                expression { return params.create_release }
            }
            steps {
                // This will drop the dev suffix if we are releasing
                // X.Y.Z.devN -> X.Y.Z
                sh "./gradlew -i bumpVersionRelease"
            }
        }

        stage ("test and build master") {
            when {
                branch 'master'
            }
            steps {
                sh "./gradlew -i build"
            }
        }

        stage ("test and build for snapshot") {
            when {
                not { branch 'master' }
            }
            steps {
                sh "./gradlew -i buildBranch -Pbranch-name=${env.BRANCH_NAME}"
            }
        }

        stage ("report on tests") {
            steps {
                junit "build/test_report.xml"
                
                cobertura autoUpdateHealth: false,
                    autoUpdateStability: false,
                    coberturaReportFile: 'build/coverage.xml',
                    failUnhealthy: false,
                    failUnstable: false,
                    maxNumberOfBuilds: 0,
                    onlyStable: false,
                    sourceEncoding: 'ASCII',
                    zoomCoverageChart: false
                

            }
        }

        stage ("publish release") {
            when {
                branch 'master'
                expression { return params.create_release }
            }
            steps {
                sh "./gradlew -i publishRelease"
                sh "./gradlew -i gitTagCommitPush"
                sh "./gradlew -i bumpVersionPostRelease gitCommitPush"
             }
        }

        stage ("publish snapshot") {
            when {  // publish any non-release branch to the snapshot repo
                not { expression { return params.create_release } }
            }
            steps {
                sh "./gradlew -i publishSnapshot"
                script {
                    def ignoreAuthors = ["jenkins", "Jenkins User", "Jenkins Builder"]
                    if (!ignoreAuthors.contains(gitAuthor()) && (env.BRANCH_NAME == "master")) {
                        sh "./gradlew -i bumpVersionDev gitCommitPush"
                    }
                }
            }
        }

    }
    post {
        always {
            this.notifyBuildOnSlack(currentBuild.result, currentBuild.previousBuild?.result)
            this.notifyAICSBitBucket(currentBuild.result)
        }
        failure {
            this.notifyEmail('failed')
        }
        unstable {
            this.notifyEmail('unstable')
        }
        fixed {
            this.notifyEmail('back to normal')
        }
    }
}

def notifyAICSBitBucket(String state) {
    // on success, result is null
    state = state ?: "SUCCESS"

    if (state == "SUCCESS" || state == "FAILURE") {
        currentBuild.result = state
    }

    notifyBitbucket commitSha1: "${GIT_COMMIT}",
                credentialsId: 'aea50792-dda8-40e4-a683-79e8c83e72a6',
                disableInprogressNotification: false,
                considerUnstableAsSuccess: true,
                ignoreUnverifiedSSLPeer: false,
                includeBuildNumberInKey: false,
                prependParentProjectKey: false,
                projectKey: 'SW',
                stashServerBaseUrl: 'https://aicsbitbucket.corp.alleninstitute.org'
}

def notifyBuildOnSlack(String buildStatus = 'STARTED', String priorStatus) {
    // build status of null means successful
    buildStatus =  buildStatus ?: 'SUCCESS'

    // Override default values based on build status
    if (buildStatus != 'SUCCESS') {
        slackSend (
                color: '#FF0000',
                message: "${buildStatus}: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})"
        )
    } else if (priorStatus != 'SUCCESS') {
        slackSend (
                color: '#00FF00',
                message: "BACK_TO_NORMAL: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})"
        )
    }
}

def notifyEmail(String status) {
    // Send email only to the commit authors since last successful build.
    // Param: status - This should be one of [ 'failed', 'unstable', 'back to normal' ]
    def subject = "JOB ${status.toUpperCase()}: ${env.JOB_NAME}"
    def title = ""
    if (status == "failed") {
        title = "<h3 style='color: #ff0000;'>Failed Job</h3>"
    }
    else if (status == "unstable") {
        title = "<h3 style='color: #cc6600;'>Unstable Job</h3>"
    }
    else {
        title = "<h3 style='color: #009933;'>Job Back To Normal</h3>"
    }
    def body = """
    ${title}
    <table>
        <tr>
            <td align="right"><b>Job:</b></td>
            <td><a href="${JOB_URL}">${env.JOB_NAME}</a></td>
        </tr>
        <tr>
            <td align="right"><b>Build:</b></td>
            <td><a href="${env.BUILD_URL}">Number [${env.BUILD_NUMBER}]</a> <a href="${env.BUILD_URL}console">(Console Output)</a></td>
        </tr>
        <tr>
            <td align="right"><b>Branch:</b></td>
            <td>${env.BRANCH_NAME}</td>
        </tr>
        <tr>
            <td align="right"><b>Git URL:</b></td>
            <td><a href="${env.GIT_URL}">${env.GIT_URL}</a></td>
        </tr>
        <tr>
            <td align="right"><b>Git Commit:</b></td>
            <td>${env.GIT_COMMIT}</td>
        </tr>
    </table>
    """
    emailext (
        subject: "${subject}",
        body: "${body}",
        recipientProviders: [
            [$class: 'DevelopersRecipientProvider'],
            [$class: 'RequesterRecipientProvider']
        ]
    )
}

def gitAuthor() {
    sh(returnStdout: true, script: 'git log -1 --format=%an').trim()
}
